"""Tests for the spreading-activation kernel and activation-driven rendering.

Contract checks (fast, no LLM, no daemon):
  • Per-field vector split exists in EMBEDDING_GROUPS
  • iter_families handles legacy + new shape
  • _compose_enriched_edge_text produces expected text
  • _allocate_budget_softmax respects minimum per node + total budget
  • _mask_node_by_field_activation masks below-threshold fields

Behavioral checks (use r3 preserved brain from N=5 run — read-only):
  • spread_activation from Hawaii seed activates NYC
  • format_surface_output_activation produces rendered output with
    "5 days" (from NYC content) and "10 days" (from Hawaii content)
  • Trace metadata contains activation_count + kernel_trace

Skipped when:
  • r3 brain isn't preserved on disk
  • embedder can't load the nomic model
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R3_BRAIN = os.path.expanduser(
    '~/AgentsContext/brain-eval-n5_full_parallel/edced276-r3')
HAS_R3 = os.path.isdir(R3_BRAIN)


# ─────────────────────────────────────────────────────────────────
# Contract tests — fast, no external state
# ─────────────────────────────────────────────────────────────────

class TestEmbeddingGroupsExtension(unittest.TestCase):
    """Part C: EMBEDDING_GROUPS has legacy + field cohorts, both intact."""

    def test_both_cohorts_present(self):
        from servers.pipeline_contract import EMBEDDING_GROUPS
        cohorts = {g.get('cohort') for g in EMBEDDING_GROUPS.values()}
        self.assertIn('legacy', cohorts)
        self.assertIn('field', cohorts)

    def test_field_cohort_has_zero_weights(self):
        """Field cohort must not participate in recall's top2_avg scoring."""
        from servers.pipeline_contract import EMBEDDING_GROUPS
        for name, cfg in EMBEDDING_GROUPS.items():
            if cfg.get('cohort') == 'field':
                self.assertEqual(cfg['weight'], 0.0,
                    f'field cohort {name} should have weight=0')

    def test_field_vector_types_stable_order(self):
        from servers.pipeline_contract import field_vector_types
        order = field_vector_types()
        self.assertEqual(order, ['title', 'content', 'situation', 'reasoning',
                                 'user_raw_quote', 'anchor_raw_quote', 'question'])

    def test_field_fallback_chain_covers_all(self):
        from servers.pipeline_contract import FIELD_VECTOR_FALLBACK, field_vector_types
        for t in field_vector_types():
            self.assertIn(t, FIELD_VECTOR_FALLBACK)


class TestIterFamiliesShape(unittest.TestCase):
    """Part 1 helper: iter_families handles both legacy list + new dict shapes."""

    def test_legacy_list_shape(self):
        from servers.scales.s2.edge_families import iter_families
        legacy = {'correction': ['corrects', 'supersedes']}
        out = list(iter_families(legacy))
        self.assertEqual(out, [('correction', ['corrects', 'supersedes'], '')])

    def test_new_dict_shape(self):
        from servers.scales.s2.edge_families import iter_families
        new = {'correction': {'members': ['corrects'], 'meaning': 'deltas'}}
        out = list(iter_families(new))
        self.assertEqual(out, [('correction', ['corrects'], 'deltas')])

    def test_mixed_shapes_coexist(self):
        """Migration-in-progress: some families migrated, others not."""
        from servers.scales.s2.edge_families import iter_families
        mixed = {
            'new': {'members': ['a'], 'meaning': 'x'},
            'old': ['b', 'c'],
        }
        out = sorted(iter_families(mixed))
        self.assertEqual(out[0], ('new', ['a'], 'x'))
        self.assertEqual(out[1], ('old', ['b', 'c'], ''))

    def test_malformed_entries_silently_skipped(self):
        from servers.scales.s2.edge_families import iter_families
        bad = {
            'good': {'members': ['x'], 'meaning': 'm'},
            'malformed': {'members': 'not_a_list'},
            'also_bad': 42,
            '__meta__': {'members': ['skip']},  # metadata prefix
        }
        out = list(iter_families(bad))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], 'good')


class TestEnrichedEdgeText(unittest.TestCase):
    """Edge text composition: target+relation+desc+family_meaning."""

    def test_composes_all_parts(self):
        from servers.scales.s1.surface_contract import _compose_enriched_edge_text
        edge = {'title': 'NYC trip', 'relation': 'contextualizes',
                'description': 'Hawaii is the contrastive frame'}
        rel_to_family = {'contextualizes': 'bridge'}
        meanings = {'bridge': 'edges where the relationship is explained'}
        text = _compose_enriched_edge_text(edge, rel_to_family, meanings)
        self.assertIn('NYC trip', text)
        self.assertIn('contextualizes', text)
        self.assertIn('Hawaii', text)
        self.assertIn('family:', text)
        self.assertIn('explained', text)

    def test_missing_parts_drop_silently(self):
        from servers.scales.s1.surface_contract import _compose_enriched_edge_text
        edge = {'title': 'Bare node', 'relation': 'related'}
        text = _compose_enriched_edge_text(edge, {}, {})
        # No description, no family meaning — just title + [relation]
        self.assertIn('Bare node', text)
        self.assertIn('[related]', text)
        self.assertNotIn('family:', text)


class TestBudgetAllocation(unittest.TestCase):
    """Softmax budget split across activated nodes."""

    def test_minimum_per_node(self):
        from servers.scales.s1.surface_contract import (
            _allocate_budget_softmax, _MIN_NODE_BUDGET_CHARS)
        acts = [0.1, 0.1, 0.1, 0.1]  # low activations
        budgets = _allocate_budget_softmax(acts, total_budget=400)
        for b in budgets:
            self.assertGreaterEqual(b, _MIN_NODE_BUDGET_CHARS,
                'every node gets at least minimum budget')

    def test_high_activation_gets_more(self):
        from servers.scales.s1.surface_contract import _allocate_budget_softmax
        # One saturated, rest low
        acts = [1.0, 0.3, 0.3, 0.3]
        budgets = _allocate_budget_softmax(acts, total_budget=2000)
        self.assertGreater(budgets[0], budgets[1],
            'saturated node should get more budget than low-activation')

    def test_uniform_activation_splits_evenly(self):
        from servers.scales.s1.surface_contract import _allocate_budget_softmax
        acts = [0.9, 0.9, 0.9, 0.9]
        budgets = _allocate_budget_softmax(acts, total_budget=2000)
        # All roughly equal (softmax on equal inputs → uniform)
        for b in budgets:
            self.assertAlmostEqual(b, budgets[0], delta=5)

    def test_empty_input(self):
        from servers.scales.s1.surface_contract import _allocate_budget_softmax
        self.assertEqual(_allocate_budget_softmax([], 1000), [])


class TestFieldMasking(unittest.TestCase):
    """Low-activation fields get masked out before render."""

    def test_low_activation_content_is_masked(self):
        from servers.scales.s1.surface_contract import _mask_node_by_field_activation
        node = {'title': 't', 'content': 'c-body', 'situation': 's-body'}
        fa = {'content': 0.1, 'situation': 0.8}  # content below threshold
        masked = _mask_node_by_field_activation(node, fa)
        self.assertEqual(masked['content'], '')
        self.assertEqual(masked['situation'], 's-body')  # high stays

    def test_metadata_fields_masked_by_field_activation(self):
        from servers.scales.s1.surface_contract import _mask_node_by_field_activation
        node = {
            'title': 't', 'content': 'c',
            '_metadata': {
                'reasoning': 'r',
                'user_raw_quote': 'tom said',
                'anchor_raw_quote': 'anchor said',
                'question': 'what?',
            }
        }
        fa = {
            'reasoning': 0.9,
            'user_raw_quote': 0.1,       # below — masks
            'anchor_raw_quote': 0.5,
            'question': 0.05,             # below — masks
        }
        masked = _mask_node_by_field_activation(node, fa)
        self.assertIn('reasoning', masked['_metadata'])
        self.assertNotIn('user_raw_quote', masked['_metadata'])
        self.assertIn('anchor_raw_quote', masked['_metadata'])
        self.assertNotIn('question', masked['_metadata'])

    def test_missing_field_activation_defaults_visible(self):
        """When a field has no activation recorded, we default to visible
        (conservative). Only explicit below-threshold causes masking."""
        from servers.scales.s1.surface_contract import _mask_node_by_field_activation
        node = {'title': 't', 'content': 'c', 'situation': 's'}
        fa = {}  # no activations — all fields should stay
        masked = _mask_node_by_field_activation(node, fa)
        self.assertEqual(masked['content'], 'c')
        self.assertEqual(masked['situation'], 's')


# ─────────────────────────────────────────────────────────────────
# Behavioral tests — need r3 preserved brain + embedder
# ─────────────────────────────────────────────────────────────────

@unittest.skipUnless(HAS_R3, 'r3 preserved brain not on disk (run N=5 first)')
class TestSpreadActivationOnR3(unittest.TestCase):
    """End-to-end: spread_activation from Hawaii activates NYC (the multi_session
    case that failed pre-redesign)."""

    @classmethod
    def setUpClass(cls):
        from servers import embedder
        if embedder._model is None:
            embedder.load_model()
        cls.tmp = tempfile.mkdtemp(prefix='test_spread_')
        for f in os.listdir(R3_BRAIN):
            shutil.copy2(os.path.join(R3_BRAIN, f), cls.tmp)
        os.environ['BRAIN_DB_DIR'] = cls.tmp
        from servers.brain import Brain
        cls.brain = Brain(db_path=os.path.join(cls.tmp, 'brain.db'))
        cls.brain.backfill_vectors(batch_size=50)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.brain.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_nyc_activates_from_hawaii_seed(self):
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '62f666c0%'").fetchone()[0]
        nyc = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '4e8acb7a%'").fetchone()[0]

        query = ("How many days did I spend in total traveling "
                 "in Hawaii and in New York City?")
        qv = np.frombuffer(embed_query(query), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        self.assertIn(nyc, result['node_activation'],
            'NYC did not activate — spreading failed for multi_session case')
        self.assertGreater(result['node_activation'][nyc], 0.5,
            f'NYC activation too weak: {result["node_activation"][nyc]}')

    def test_community_bridge_activates_on_relevant_query(self):
        """Community activates when query semantics match community meaning.
        Query-dependent by design: 'travel style' triggers the Solo Travel
        Profile community naturally; a narrow day-counting query wouldn't."""
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '62f666c0%'").fetchone()[0]
        community = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '3d25ca4c%'").fetchone()[0]

        query = "What's my travel style?"
        qv = np.frombuffer(embed_query(query), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        self.assertIn(community, result['node_activation'],
            'community bridge did not activate for a style-relevant query')

    def test_field_activations_populated_for_activated_nodes(self):
        """Each activated node should have per-field activations for rendering."""
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '62f666c0%'").fetchone()[0]
        qv = np.frombuffer(embed_query("Hawaii trip"), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        # At least the seed should have field activations
        self.assertIn(hawaii, result['field_activation'])
        self.assertGreater(len(result['field_activation'][hawaii]), 0,
            'seed has no field activations')

    def test_kernel_trace_present(self):
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import spread_activation
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '62f666c0%'").fetchone()[0]
        qv = np.frombuffer(embed_query("travel"), dtype=np.float32)

        result = spread_activation([hawaii], qv, self.brain)

        self.assertGreater(len(result['trace']), 0,
            'no kernel trace entries')
        first_step = result['trace'][0]
        for key in ('step', 'new_nodes', 'edges_considered',
                    'edges_transmitted', 'max_act'):
            self.assertIn(key, first_step,
                f'trace step missing key: {key}')


@unittest.skipUnless(HAS_R3, 'r3 preserved brain not on disk (run N=5 first)')
class TestActivationRenderingOnR3(unittest.TestCase):
    """format_surface_output_activation produces the 5-day/10-day composition
    content that the old pipeline missed."""

    @classmethod
    def setUpClass(cls):
        from servers import embedder
        if embedder._model is None:
            embedder.load_model()
        cls.tmp = tempfile.mkdtemp(prefix='test_render_')
        for f in os.listdir(R3_BRAIN):
            shutil.copy2(os.path.join(R3_BRAIN, f), cls.tmp)
        os.environ['BRAIN_DB_DIR'] = cls.tmp
        from servers.brain import Brain
        cls.brain = Brain(db_path=os.path.join(cls.tmp, 'brain.db'))
        cls.brain.backfill_vectors(batch_size=50)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.brain.close()
        except Exception:
            pass
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_nyc_content_visible_in_rendered_output(self):
        """The r3 failing case: full pipeline renders NYC's 5-day fact."""
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import (
            spread_activation, format_surface_output_activation)
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '62f666c0%'").fetchone()[0]
        nyc = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '4e8acb7a%'").fetchone()[0]

        query = ("How many days did I spend in total traveling in "
                 "Hawaii and in New York City?")
        qv = np.frombuffer(embed_query(query), dtype=np.float32)

        activation_result = spread_activation([hawaii], qv, self.brain)
        all_ids = list(activation_result['node_activation'].keys())
        rich_nodes = self.brain.get_node(all_ids)

        rendered = format_surface_output_activation(
            node_activation=activation_result['node_activation'],
            field_activation=activation_result['field_activation'],
            rich_nodes=rich_nodes,
            selected_why={hawaii: 'seed'},
            query_vec=qv,
            brain=self.brain,
            session=None,
            total_budget=4000,
        )

        self.assertGreater(len(rendered), 500,
            'rendered output too small')
        has_nyc_ref = 'NYC' in rendered or 'New York' in rendered
        self.assertTrue(has_nyc_ref, 'NYC not mentioned in rendered output')
        has_five = '5 days' in rendered or '5-day' in rendered or 'five days' in rendered
        self.assertTrue(has_five, 'NYC 5-day fact not in rendered output')
        has_ten = '10 days' in rendered or '10-day' in rendered or 'ten days' in rendered
        self.assertTrue(has_ten, 'Hawaii 10-day fact not in rendered output')

    def test_rendered_output_stays_under_budget(self):
        from servers.embedder import embed_query
        from servers.scales.s1.surface_contract import (
            spread_activation, format_surface_output_activation)
        import numpy as np

        hawaii = self.brain.conn.execute(
            "SELECT id FROM nodes WHERE id LIKE '62f666c0%'").fetchone()[0]
        qv = np.frombuffer(embed_query("travel"), dtype=np.float32)

        act = spread_activation([hawaii], qv, self.brain)
        rich = self.brain.get_node(list(act['node_activation'].keys()))

        budget = 2000
        rendered = format_surface_output_activation(
            node_activation=act['node_activation'],
            field_activation=act['field_activation'],
            rich_nodes=rich,
            selected_why={hawaii: 'seed'},
            query_vec=qv,
            brain=self.brain, session=None,
            total_budget=budget,
        )
        # Allow overrun: MIN_NODE_BUDGET ensures each activated node gets
        # ~150 chars of content minimum; with 10+ nodes that adds up. What
        # we guard against is runaway output, not per-node slack.
        self.assertLess(len(rendered), budget * 2,
            f'rendered output {len(rendered)} ballooned beyond 2x budget {budget}')


if __name__ == '__main__':
    unittest.main()
