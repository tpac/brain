#!/usr/bin/env python3
"""
brain — Recall Quality Tests

Tests the scoring pipeline that determines what the brain surfaces:
- Scoring weights (relevance, frequency, emotion, locked decay)
- Intent detection and type boosting
- Dampening (hub, type, confidence, project filtering)
- Spreading activation (multi-hop, decay, max hops)
- TF-IDF recall (fallback path, stopwords, cosine similarity)

Run: python -m unittest tests.test_recall_quality -v
"""

import sys
import os
import unittest
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase


# ═══════════════════════════════════════════════════════════════
# TestScoringWeights — verify that the scoring formula respects
# the intended priority of relevance, frequency, emotion, and locked status
# ═══════════════════════════════════════════════════════════════

class TestScoringWeights(BrainTestBase):
    """Verify that scoring weights produce correct ranking behavior."""

    def test_relevance_dominates(self):
        """A node highly relevant to the query should outrank a recent but irrelevant node.

        Creates a Clerk authentication decision (matches query) and a database
        migration timeline decision (does not match). Even though both are fresh,
        the Clerk node should rank first because relevance weight (0.35) is the
        highest single factor and keyword overlap is strong.
        """
        self.brain.remember(
            type='decision',
            title='Authentication: adopt Clerk with magic links for passwordless login',
            content='We evaluated Auth0, Firebase Auth, and Clerk. Clerk won because its magic link flow '
                    'eliminates password fatigue, integrates with Next.js middleware out of the box, '
                    'and supports organization-scoped sessions for our multi-tenant architecture.',
            keywords='auth clerk login magic-link passwordless authentication nextjs middleware'
        )
        self.brain.remember(
            type='decision',
            title='Database migration timeline for Q2 schema overhaul',
            content='The PostgreSQL migration from v12 to v16 is scheduled for late April. We need to '
                    'coordinate with the DevOps team on blue-green deployment and ensure pgbouncer '
                    'connection pooling is configured before the cutover window.',
            keywords='database migration postgresql devops deployment pgbouncer schema'
        )
        self.brain.save()

        results = self.brain.recall('auth login clerk', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 1, 'Should find at least one result')
        self.assertIn('Clerk', result_list[0]['title'],
                      f'Clerk node should rank first, got: {result_list[0]["title"]}')

    def test_locked_never_decays(self):
        """A locked rule node should appear in recall even if its last_accessed is 60 days old.

        Locked nodes have infinite half-life and skip decay calculations entirely.
        This verifies the locked code path in the scoring pipeline.
        """
        r = self.brain.remember(
            type='rule',
            title='Never commit secrets to version control',
            content='All API keys, tokens, and credentials must live in environment variables or '
                    'a secrets manager (AWS SSM Parameter Store or 1Password CLI). Pre-commit hooks '
                    'with detect-secrets must be enabled on every repository. Violations trigger '
                    'immediate secret rotation and a post-mortem.',
            keywords='secrets security api-keys credentials pre-commit detect-secrets',
            locked=True
        )
        self.brain.save()

        # Manually backdate last_accessed to 60 days ago
        sixty_days_ago = (datetime.utcnow() - timedelta(days=60)).isoformat() + 'Z'
        self.brain.conn.execute(
            'UPDATE nodes SET last_accessed = ? WHERE id = ?',
            (sixty_days_ago, r['id'])
        )
        self.brain.conn.commit()

        results = self.brain.recall('secrets credentials api keys security', limit=10)
        result_list = results.get('results', [])
        found_ids = [n['id'] for n in result_list]
        self.assertIn(r['id'], found_ids,
                      'Locked node should still appear despite being 60 days old')


# ═══════════════════════════════════════════════════════════════
# TestDampening — verify that hub nodes, low-value types,
# and low-confidence nodes are appropriately penalized
# ═══════════════════════════════════════════════════════════════

class TestDampening(BrainTestBase):
    """Verify that dampening mechanisms reduce noise from hubs and low-signal types."""

    # test_hub_dampening removed 2026-08-13: it drove `brain.recall()` and
    # asserted final ordering, but hub dampening (`relevance *= threshold /
    # edge_count`) only touches the keyword channel inside `_keyword_recall`
    # — under laf_v1 the field score dominates the blend, so the assertion
    # never described this path. Hub suppression on the main path is an open
    # LAF lane question (brain 7e9e36a7), not a regression this pins.

    def test_type_dampening(self):
        """A 'project' node should rank lower than a 'decision' node with similar content.

        Project and person nodes get a 0.5x dampening factor because they tend to
        match too many queries without providing actionable information.
        """
        self.brain.remember(
            type='project',
            title='Webhook infrastructure overhaul',
            content='The webhook system needs to be redesigned to support retry logic with '
                    'exponential backoff, dead letter queues, and payload signature verification. '
                    'Current implementation drops events silently on 5xx responses.',
            keywords='webhook infrastructure retry backoff dead-letter-queue signature'
        )
        self.brain.remember(
            type='decision',
            title='Webhook retry: use exponential backoff with jitter',
            content='After evaluating linear retry, exponential backoff, and fibonacci backoff, '
                    'we chose exponential with jitter. Base delay is 1 second, max 5 minutes, '
                    'jitter range is 0-30% of the delay. Dead events go to SQS DLQ after 5 retries.',
            keywords='webhook retry exponential-backoff jitter dead-letter-queue sqs'
        )
        self.brain.save()

        results = self.brain.recall('webhook retry backoff strategy', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 2,
                        f'Should find at least 2 results, got {len(result_list)}')
        # Decision node should rank above project node
        self.assertEqual(result_list[0]['type'], 'decision',
                         f'Decision should rank first (type dampening), got type={result_list[0]["type"]}')

    def test_project_filtering(self):
        """Project is kv provenance (2026-07-03): the dict filter hard-scopes
        by it, replacing the removed recall(project=) soft-sort. The soft
        version is the LAF proj lane (gain-dialed, session-derived —
        tests/test_recall_laf.py::TestProjLane)."""
        a = self.brain.remember(
            type='decision',
            title='Alpha: use GraphQL for the dashboard API layer',
            content='The alpha project dashboard needs flexible queries for widgets. GraphQL lets '
                    'the frontend request exactly the fields it needs without over-fetching. We use '
                    'Apollo Server with DataLoader for N+1 query prevention.',
            project='alpha'
        )
        self.brain.remember(
            type='decision',
            title='Beta: use REST for the public developer API',
            content='The beta project public API must be REST because our developer audience expects '
                    'standard HTTP methods, predictable URLs, and OpenAPI documentation. GraphQL would '
                    'add unnecessary complexity for third-party consumers.',
            project='beta'
        )
        self.brain.save()

        # project rides **extra_fields into node_metadata_kv, and get_node
        # promotes it back onto the payload (kv wins over the legacy column)
        node = self.brain.get_node(a['id'])
        self.assertEqual(node.get('project'), 'alpha')
        self.assertEqual(
            self.brain._meta_kv.get_field(a['id'], 'project'), 'alpha')

        # dict filter routes project to the kv lookup (left _NODE_COLUMNS)
        results = self.brain.recall(
            'API design decisions', limit=5,
            filter={'project': {'equals': 'alpha'}})
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 1,
                        f'Should find at least 1 result, got {len(result_list)}')
        for n in result_list:
            self.assertEqual(
                self.brain._meta_kv.get_field(n['id'], 'project'), 'alpha',
                'project filter must hard-scope to alpha nodes only')


# Coverage moved 2026-04-26 to tests/test_spread_activation.py.
# The old TestSpreadingActivation class called brain.spread_activation([seeds])
# — a method-on-Brain API that no longer exists. The kernel was rewritten as
# a query-aware module function `servers.scales.s1.surface_contract.spread_activation(
# seed_ids, query_vec, brain, prior_vecs=None)`. The new contract requires a
# query embedding (activation originates from max field-cosine with query),
# making the old "spread without query" tests not translatable. The new
# test_spread_activation.py covers the full kernel: edge enrichment text
# composition, softmax budget allocation, field activation masking, and
# end-to-end Hawaii→NYC seed propagation. The replacement tests assert
# stricter properties than the originals (edge text compositions, budget
# distributions, field masks) — not weaker.

class TestSpreadingActivationCoverageMoved(BrainTestBase):
    """Sentinel — placeholder pointing to the new home (test_spread_activation.py)."""

    def test_kernel_module_importable(self):
        """The new spread_activation lives in surface_contract — verify the move landed."""
        from servers.scales.s1.surface_contract import spread_activation
        self.assertTrue(callable(spread_activation))




# ═══════════════════════════════════════════════════════════════
# TestEmbedderDownNoFallback — embedder unavailable/failed must yield EMPTY
# results + a reported condition, never a silent keyword substitute
# ═══════════════════════════════════════════════════════════════

class TestEmbedderDownNoFallback(BrainTestBase):
    """Recall's no-embedding exits: empty results, consistent shape, loud report.

    Guards the no-fallback contract — a future
    edit that re-introduces a silent keyword substitute, drops the _log_error
    report, or diverges the empty-result shape fails here."""

    def test_embedder_down_returns_empty_and_reports(self):
        from unittest.mock import patch
        from servers import embedder

        original_model = embedder._model
        original_loaded = embedder.stats['model_loaded']
        embedder._model = None
        embedder.stats['model_loaded'] = False
        try:
            with patch.object(self.brain, '_log_error') as log_spy:
                results = self.brain.recall('embedder down empty contract', limit=5)
        finally:
            embedder._model = original_model
            embedder.stats['model_loaded'] = original_loaded

        self.assertEqual(results.get('results'), [],
                         'no keyword substitute — results must be EMPTY')
        self.assertEqual(results.get('_recall_mode'), 'embedder_unavailable')
        stats = results.get('_embedding_stats')
        self.assertTrue(stats, '_embedding_stats must be present so the MCP '
                               'footer renders the failure mode inline')
        self.assertFalse(stats.get('embedder_ready'))
        sources = [c.args[0] for c in log_spy.call_args_list]
        self.assertIn('recall_embedder_unavailable', sources,
                      'the condition must be REPORTED via _log_error')

    def test_embed_failure_returns_empty_and_reports(self):
        from unittest.mock import patch
        from servers import embedder

        with patch.object(embedder, 'embed_query',
                          side_effect=RuntimeError('embed boom')), \
             patch.object(self.brain, '_log_error') as log_spy:
            results = self.brain.recall('embed failure empty contract', limit=5)

        self.assertEqual(results.get('results'), [],
                         'no keyword substitute — results must be EMPTY')
        self.assertEqual(results.get('_recall_mode'), 'embed_failed')
        self.assertTrue(results.get('_embedding_stats'),
                        'embed_failed must carry _embedding_stats too — '
                        'same shape as embedder_unavailable')
        sources = [c.args[0] for c in log_spy.call_args_list]
        self.assertIn('recall_embed_failed', sources,
                      'the condition must be REPORTED via _log_error')


# ═══════════════════════════════════════════════════════════════
# TestTFIDFRecall — verify TF-IDF stopword filtering and cosine similarity
# scoring (the keyword-net internals)
# ═══════════════════════════════════════════════════════════════

class TestTFIDFRecall(BrainTestBase):
    """Verify the TF-IDF keyword-net internals (tokenizer, cosine scoring)."""

    def test_stopwords_filtered(self):
        """Common English stopwords should NOT appear in the node_vectors TF-IDF table.

        The tokenizer must strip words like 'the', 'is', 'at', etc. before indexing.
        """
        n = self.brain.remember(
            type='lesson',
            title='The quick brown fox jumps over the lazy dog',
            content='This is a classic pangram that contains every letter of the English alphabet. '
                    'It has been used for testing typewriters and fonts since the late 1800s.',
            keywords='pangram typewriter fonts alphabet classic testing'
        )
        self.brain.save()

        # Check that 'the' (a stopword) is NOT in node_vectors
        cursor = self.brain.conn.execute(
            "SELECT term FROM node_vectors WHERE node_id = ? AND term = 'the'",
            (n['id'],)
        )
        row = cursor.fetchone()
        self.assertIsNone(row, '"the" is a stopword and should not be in node_vectors')

        # Verify that a non-stopword IS present
        cursor = self.brain.conn.execute(
            "SELECT term FROM node_vectors WHERE node_id = ? AND term = 'quick'",
            (n['id'],)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row, '"quick" should be indexed in node_vectors')

    def test_tfidf_cosine_similarity(self):
        """Two nodes with overlapping terms should have higher TF-IDF similarity than disjoint nodes.

        Creates three nodes: two about caching (overlapping terms) and one about
        authentication (disjoint). TF-IDF scores for a caching query should be
        higher for the caching nodes.
        """
        n1 = self.brain.remember(
            type='decision',
            title='Redis caching strategy for user session data',
            content='User sessions are cached in Redis with a 30-minute TTL. Cache-aside pattern: '
                    'check cache first, fall through to PostgreSQL, populate cache on miss. Cache '
                    'invalidation happens on password change and role modification.',
            keywords='redis cache session ttl cache-aside postgresql invalidation'
        )
        n2 = self.brain.remember(
            type='decision',
            title='CDN caching rules for static assets and API responses',
            content='Static assets get Cache-Control: max-age=31536000 with content-hash filenames. '
                    'API responses use stale-while-revalidate with 60-second max-age. The CDN cache '
                    'is purged on deployment via Cloudflare API.',
            keywords='cdn cache static-assets cache-control stale-while-revalidate cloudflare'
        )
        n3 = self.brain.remember(
            type='decision',
            title='OAuth2 implementation for third-party integrations',
            content='Third-party integrations authenticate via OAuth2 authorization code flow. '
                    'Access tokens expire after 1 hour, refresh tokens after 30 days. Token storage '
                    'uses encrypted columns in PostgreSQL.',
            keywords='oauth2 authentication tokens refresh-token authorization integration'
        )
        self.brain.save()

        # Get TF-IDF scores for a caching-related query
        query_terms = self.brain._tfidf_tokenize('redis caching strategy session data')
        scores = self.brain._batch_tfidf_scores(query_terms, [n1['id'], n2['id'], n3['id']])

        score_n1 = scores.get(n1['id'], 0)
        score_n2 = scores.get(n2['id'], 0)
        score_n3 = scores.get(n3['id'], 0)

        # Both caching nodes should score higher than the OAuth node
        self.assertGreater(score_n1, score_n3,
                           f'Redis caching node ({score_n1}) should score higher than OAuth node ({score_n3})')
        self.assertGreater(score_n2, score_n3,
                           f'CDN caching node ({score_n2}) should score higher than OAuth node ({score_n3})')
        # The direct redis+session match should score highest
        self.assertGreater(score_n1, score_n2,
                           f'Redis session caching ({score_n1}) should score higher than CDN caching ({score_n2})')


if __name__ == '__main__':
    unittest.main()
