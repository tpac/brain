"""LAF v1 recall variant — engine units, DAL door, flag gating (§19 P1).

Component tests (no embedder): lane math, config overlay, id resolution, the
uncapped event_vector_rows DAL door (incl. incremental `since`).
Integration smoke (real embedder + production-copy brain via IsolatedBrain):
flag-on recall returns laf-scored results end-to-end; flag-off same brain is
champion-shaped. Skipped when no production DB is available.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase                      # noqa: E402
from servers.recall_laf import (                                     # noqa: E402
    DEFAULT_CONFIG, LafV1Engine, MAXSIM_VIEWS, _unit, _zscore, idf_scores,
)


class TestLafEngineUnits(BrainTestBase):
    needs_embedder = False

    def test_maxsim_views_from_contract(self):
        # The views derive from EMBEDDING_GROUPS — the live cohort, no dead names.
        self.assertIn('_primary', MAXSIM_VIEWS)
        self.assertIn('title', MAXSIM_VIEWS)
        self.assertNotIn('_situation', MAXSIM_VIEWS)  # situation is its own lane

    def test_unit_normalizes_and_rejects_zero(self):
        v = _unit(np.array([3.0, 4.0], dtype=np.float32).tobytes())
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=5)
        self.assertIsNone(_unit(np.zeros(2, dtype=np.float32).tobytes()))
        self.assertIsNone(_unit(None))

    def test_zscore_unit_variance_nan_safe_and_mask(self):
        x = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        z = _zscore(x, 5)
        finite = z[np.isfinite(x)]
        self.assertAlmostEqual(float(finite.std()), 1.0, places=6)
        self.assertEqual(z[3], 0.0)          # NaN slot → neutral, not poison
        # mask form (the laf_metrics.zscore delegation shape): masked-out
        # entries are neutral AND excluded from the statistics
        mask = np.array([True, True, True, True, False])
        zm = _zscore(x, 5, mask=mask)
        self.assertEqual(zm[4], 0.0)
        self.assertAlmostEqual(float(zm[[0, 1, 2]].std()), 1.0, places=6)

    def test_config_defaults_kstore_overlay_and_gain_per_field(self):
        # one gain key per registry field — the P3 fit surface
        for field in ('maxsim', 'pick', 'enc', 'idf', 'sit'):
            self.assertIn('gain_' + field, DEFAULT_CONFIG)

        class FakeBrain:
            def get_interaction_config(self, name):
                assert name == 'recall_laf'
                return {'gain_pick': 0.9, 'unknown_key': 'ignored'}

        cfg = LafV1Engine().config(FakeBrain())
        self.assertEqual(cfg['gain_pick'], 0.9)          # override applied
        self.assertEqual(cfg['gain_enc'], DEFAULT_CONFIG['gain_enc'])
        self.assertNotIn('unknown_key', cfg)             # unknown keys dropped

        class BrokenBrain:
            def get_interaction_config(self, name):
                raise RuntimeError('K-store down')
        # fresh engine (config is TTL-cached per engine); broken K-store →
        # defaults, and the failure is logged loud when _log_error exists
        logged = []
        BrokenBrain._log_error = lambda self, *a: logged.append(a)
        self.assertEqual(LafV1Engine().config(BrokenBrain()), DEFAULT_CONFIG)
        self.assertEqual(logged[0][0], 'recall_laf_config')

    def test_idf_scores_rare_token_wins(self):
        # 3 titles: row 0 has the rare token, rows 1-2 share a flood token
        title_tok = {0: frozenset({'spread_activation', 'recall'}),
                     1: frozenset({'recall', 'notes'}),
                     2: frozenset({'recall'})}
        title_df = {'spread_activation': 1, 'recall': 3, 'notes': 1}
        vec = idf_scores('how does spread_activation recall work',
                         title_tok, title_df, 3)
        self.assertGreater(vec[0], vec[1])   # rare-token title dominates
        # 'recall' hits ALL titles → idf log((3+1)/(3+1)) = 0: flood terms are
        # flattened to exactly nothing (the idf2 design), so rows 1-2 score 0
        self.assertEqual(vec[1], 0.0)
        self.assertEqual(vec[2], 0.0)
        # stopwords/short tokens contribute nothing
        self.assertTrue(np.all(idf_scores('a of to', title_tok, title_df, 3) == 0.0))

    def test_resolve_short_and_full(self):
        eng = LafV1Engine()
        eng._idx = {'abcdef1234567890': 0}
        eng._short = {'abcdef12': 0}
        self.assertEqual(eng._resolve('abcdef1234567890'), 0)
        self.assertEqual(eng._resolve('abcdef12'), 0)
        self.assertIsNone(eng._resolve('ffffffff'))

    def test_event_vector_rows_uncapped_and_incremental(self):
        dal = self.brain._trace_dal
        vec = np.ones(4, dtype=np.float32).tobytes()
        for i in range(7):
            ts = '2026-01-0%dT00:00:00+00:00' % (i + 1)
            tid = 'trace-%d' % i
            dal.conn.execute(
                "INSERT INTO trace_events (id, chain_id, scale, event_type,"
                " ref_type, ref_id, summary, metadata, session_id, created_at)"
                " VALUES (?, ?, 's0', 'observation', 'user_message',"
                " ?, 's', '{}', 'sess', ?)",
                (tid, 's0-deadbeef-%d' % i, 'r%d' % i, ts))
            dal.conn.execute(
                'INSERT INTO trace_embeddings (trace_id, vector, model)'
                ' VALUES (?, ?, ?)', (tid, vec, 'test'))
        dal.conn.commit()
        rows = dal.event_vector_rows(scale='s0', ref_types=['user_message'])
        self.assertEqual(len(rows), 7)                      # uncapped, all rows
        self.assertEqual(rows[0][2][:10], '2026-01-01')     # ASC order
        # incremental: since the 5th row's timestamp → only rows 6-7
        newer = dal.event_vector_rows(scale='s0', ref_types=['user_message'],
                                      since=rows[4][2])
        self.assertEqual(len(newer), 2)
        # ref_type filter honored
        self.assertEqual(dal.event_vector_rows(scale='s0',
                                               ref_types=['tool_result']), [])


@unittest.skipUnless(
    os.path.exists(os.path.join(os.path.expanduser('~'),
                                'AgentsContext', 'brain', 'brain.db'))
    or os.environ.get('BRAIN_DB_DIR'),
    'integration smoke needs a production brain copy')
class TestLafFlagIntegration(unittest.TestCase):
    """End-to-end: flag on → laf-scored recall; flag off → champion shape.

    Runs on an IsolatedBrain copy (never the live DB) with the real embedder.
    """

    @classmethod
    def setUpClass(cls):
        from tests.isolated_brain import IsolatedBrain
        cls._env = IsolatedBrain()
        cls._env.__enter__()
        cls.brain = cls._env.brain

    @classmethod
    def tearDownClass(cls):
        cls._env.__exit__(None, None, None)

    def _recall(self, variant):
        prev = os.environ.get('BRAIN_RECALL_VARIANT')
        try:
            if variant:
                os.environ['BRAIN_RECALL_VARIANT'] = variant
            else:
                os.environ.pop('BRAIN_RECALL_VARIANT', None)
            # distinct queries per variant → the 5s result cache can't cross paths
            return self.brain.recall(
                query='recall latency measurement %s' % (variant or 'champion'),
                limit=10)
        finally:
            if prev is None:
                os.environ.pop('BRAIN_RECALL_VARIANT', None)
            else:
                os.environ['BRAIN_RECALL_VARIANT'] = prev

    def test_flag_off_is_champion(self):
        res = self._recall(None)
        self.assertEqual(res['_recall_mode'], 'embeddings_first')
        self.assertFalse(any(r.get('_discovery') == 'laf_v1'
                             for r in res['results']))

    def test_flag_on_scores_via_field(self):
        res = self._recall('laf_v1')
        self.assertEqual(res['_recall_mode'], 'laf_v1')
        self.assertTrue(res['results'], 'laf recall returned nothing')
        for r in res['results']:
            self.assertEqual(r.get('_discovery'), 'laf_v1')
            # the (0,1) range contract the champion's floors/boosts assume
            self.assertGreater(r['effective_activation'], 0.0)
            self.assertLessEqual(r['effective_activation'], 1.0)
        # per-candidate field telemetry rides the result (P2 walker feed)
        self.assertIn('_laf_fields', res)
        some_fields = next(iter(res['_laf_fields'].values()))
        self.assertEqual(set(some_fields), {'maxsim', 'pick', 'enc', 'idf', 'sit'})
        # engine cached on the brain, matrices + trace blocks resident
        eng = self.brain._laf_engine
        self.assertGreater(eng._n, 0)
        self.assertTrue(eng._tr_blocks)
        # laf results visible in the source breakdown telemetry
        self.assertGreater(
            res['_retrieval_stats']['source_breakdown'].get('laf_v1', 0), 0)

    def test_unknown_variant_falls_back(self):
        res = self._recall('nonexistent_variant')
        self.assertEqual(res['_recall_mode'], 'embeddings_first')


if __name__ == '__main__':
    unittest.main()
