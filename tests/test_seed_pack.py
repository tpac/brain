"""Seed-pack integrity — the baby-brain bootstrap must not drift.

Static checks on SEED_NODES / SEED_EDGES (no brain needed, fast). They guard the
two drifts found 2026-06-07: banned `related`/empty edge relations, and the
v28-dropped `keywords` field lingering on seed nodes. The seed has no other test
coverage, which is exactly why those drifted unnoticed — this is the floor.
"""
import unittest

from servers.seed_pack import SEED_NODES, SEED_EDGES

# Relations the rest of the system forbids: they carry zero information and
# pollute the activation kernel (the encoder is explicitly told NEVER to emit
# them). A fresh brain must not bootstrap with them.
BANNED_RELATIONS = {'related', 'related_to', ''}

# Fields dropped from the schema that must not appear on seed nodes — they would
# route to node_metadata_kv as dead metadata on every fresh install.
DEAD_NODE_FIELDS = {'keywords'}


class TestSeedPackIntegrity(unittest.TestCase):
    def test_no_banned_edge_relations(self):
        """SEED_EDGES must use meaningful relations — never related/related_to/empty."""
        offenders = [(e.get('source'), e.get('target'), e.get('relation'))
                     for e in SEED_EDGES
                     if (e.get('relation') or '').strip() in BANNED_RELATIONS]
        self.assertEqual(offenders, [],
                         "SEED_EDGES use banned/empty relations: %r" % offenders)

    def test_every_edge_has_relation_and_description(self):
        """Every seed edge needs a relation AND a why (description) — the why is
        embedded for recall, so a typed-but-undescribed edge is half-blind."""
        for e in SEED_EDGES:
            pair = (e.get('source'), e.get('target'))
            self.assertTrue((e.get('relation') or '').strip(),
                            "edge %r missing relation" % (pair,))
            self.assertTrue((e.get('description') or '').strip(),
                            "edge %r missing description" % (pair,))

    def test_no_dead_fields_on_seed_nodes(self):
        """Seed nodes must not carry schema-dropped fields (e.g. v28 `keywords`)."""
        for n in SEED_NODES:
            dead = DEAD_NODE_FIELDS & set(n.keys())
            self.assertEqual(dead, set(),
                             "seed node %r carries dead field(s) %r" % (n.get('slug'), dead))


if __name__ == '__main__':
    unittest.main()
