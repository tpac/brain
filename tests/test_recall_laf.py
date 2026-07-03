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


class TestProjLane(BrainTestBase):
    """proj lane: session-project provenance as a gain-dialed activation field.

    Design rules pinned here (2026-07-03):
      - missing data is NEUTRAL (NaN, excluded from z-stats) — the sit-lane
        zero-fill lesson as a contract;
      - no session project → the whole lane is inert;
      - gain_proj defaults to 0.0 → wired-but-contributing-nothing until a
        measured gain is registered (the gate corpus is single-project and
        cannot see this lane yet).
    """

    def _engine_with_projects(self):
        """Fresh brain + 3 embedded nodes (alpha / beta / no project)."""
        a = self.brain.remember(
            type='decision', title='Alpha decision on caching strategy',
            content='We cache aggressively at the edge for the dashboard.',
            project='alpha')
        b = self.brain.remember(
            type='decision', title='Beta decision on caching strategy',
            content='We cache conservatively at the origin for the API.',
            project='beta')
        c = self.brain.remember(
            type='decision', title='General decision on caching strategy',
            content='Caching should always be measured before tuning.')
        # third project-carrying node: _zscore's small-sample guard needs >2
        # finite entries or the lane (correctly) stays silent
        self.brain.remember(
            type='decision', title='Gamma decision on caching strategy',
            content='Gamma project caches nothing and recomputes on demand.',
            project='gamma')
        from servers.recall_laf import get_engine
        return get_engine(self.brain), a['id'], b['id'], c['id']

    def test_project_field_semantics(self):
        # unit: same-project 1.0, cross-project 0.0, no-project NaN,
        # no-session-project → all NaN (inert)
        eng = LafV1Engine()
        eng._proj_rows = np.array([0, 1], dtype=np.int64)   # row 2 carries none
        eng._proj_vals = np.array(['alpha', 'beta'], dtype=object)
        vec = eng._project_field('alpha', 3)
        assert vec[0] == 1.0 and vec[1] == 0.0 and np.isnan(vec[2])
        assert np.isnan(eng._project_field('', 3)).all()
        assert np.isnan(eng._project_field(None, 3)).all()

    def test_sit_lane_preserves_nan_for_vectorless_nodes(self):
        # REGRESSION (the sit-lane zero-fill bug): a node with no _situation
        # vector must reach the fusion as NaN — a real 0.0 z-scores ~10σ below
        # the corpus mean and buries the node. Pins the lane contract stated
        # in _fields' docstring for the lane that originally violated it.
        import servers.embedder as embedder
        from servers.recall_laf import get_engine, _unit as _u
        node = self.brain.remember(
            type='lesson', title='Node without a situation field',
            content='This node deliberately has no situation.')
        eng = get_engine(self.brain)
        qv = _u(embedder.embed_query('anything at all'))
        with eng._lock:
            eng._refresh_matrices(self.brain, None)
            nk = eng._refresh_titles(self.brain)
            eng._refresh_projects(self.brain, node_key=nk)
            eng._refresh_traces(self.brain)
            fields = eng._fields(self.brain, 'anything at all', qv,
                                 eng.config(self.brain), eng._n)
        row = eng._idx[node['id']]
        assert np.isnan(fields['sit'][row]), \
            'missing situation vector must be NaN in the sit lane, never 0.0'

    def test_full_matrix_build_resets_proj_arrays(self):
        # REGRESSION: proj arrays are row-keyed; a full matrix rebuild
        # reindexes rows, so the arrays must be force-reset like the titles
        # cache — stale indices would label the WRONG nodes.
        eng, *_ = self._engine_with_projects()
        import servers.embedder as embedder
        from servers.recall_laf import _unit as _u
        qv = _u(embedder.embed_query('caching'))
        eng.scores(self.brain, 'caching', qv, session_project='alpha')
        assert len(eng._proj_rows)                  # populated by the refresh
        with eng._lock:
            eng._full_matrix_build(self.brain, None)
        assert len(eng._proj_rows) == 0 and eng._proj_key is None

    def test_default_gain_zero_is_bit_identical(self):
        import servers.embedder as embedder
        eng, *_ = self._engine_with_projects()
        qv = embedder.embed_query('caching strategy decision')
        base, _ = eng.scores(self.brain, 'caching strategy decision', qv)
        with_proj, _ = eng.scores(self.brain, 'caching strategy decision', qv,
                                  session_project='alpha')
        assert base == with_proj      # gain 0 → lane wired, zero contribution

    def test_nonzero_gain_boosts_same_inhibits_cross_neutral_missing(self):
        import time as _time
        import servers.embedder as embedder
        eng, aid, bid, cid = self._engine_with_projects()
        # Isolate the lane: all gains 0 except proj → score IS the proj z.
        eng._cfg = {**DEFAULT_CONFIG,
                    'gain_maxsim': 0.0, 'gain_pick': 0.0, 'gain_enc': 0.0,
                    'gain_idf': 0.0, 'gain_sit': 0.0, 'gain_proj': 1.0}
        eng._cfg_ts = _time.monotonic()
        qv = embedder.embed_query('caching strategy decision')
        s, _ = eng.scores(self.brain, 'caching strategy decision', qv,
                          session_project='alpha')
        # z over finite {1.0, 0.0}: alpha +1σ, beta −1σ, no-project excluded → 0
        assert s[aid] > s[cid] > s[bid]
        # inert without a session project: everyone z=0 → equal scores
        s0, _ = eng.scores(self.brain, 'caching strategy decision', qv)
        assert s0[aid] == s0[bid] == s0[cid]

    def test_project_rows_kv_wins_over_column(self):
        # migration-proof read: node_metadata_kv['project'] beats the legacy
        # column once the migration lands, with no engine change.
        _, aid, _, cid = self._engine_with_projects()
        self.brain._meta_kv.set_many(aid, {'project': 'kv-project'})
        rows = dict(self.brain._nodes.project_rows())
        assert rows[aid] == 'kv-project'   # kv overrides column 'alpha'
        assert cid not in rows             # no project anywhere → absent


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
        self.assertEqual(set(some_fields),
                         {'maxsim', 'pick', 'enc', 'idf', 'sit', 'proj'})
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
