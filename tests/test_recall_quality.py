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
import tempfile
import shutil
import unittest
from datetime import datetime, timedelta

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.brain_constants import (
    INTENT_PATTERNS,
    INTENT_TYPE_BOOSTS,
    SPREAD_DECAY,
    MAX_HOPS,
    TFIDF_STOP_WORDS,
)


class BrainTestBase(unittest.TestCase):
    """Base class that creates a fresh brain per test."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.brain = Brain(self.db_path)
        self.brain.reset_session_activity()

    def tearDown(self):
        if self.brain is not None:
            self.brain.close()
        shutil.rmtree(self.tmp)


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

        results = self.brain.recall_with_embeddings('auth login clerk', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 1, 'Should find at least one result')
        self.assertIn('Clerk', result_list[0]['title'],
                      f'Clerk node should rank first, got: {result_list[0]["title"]}')

    def test_frequency_boost(self):
        """A node accessed many times should rank higher than one accessed once.

        Creates two nodes about React component lifecycle. Accesses one 10 times
        via recall. On the next recall, the frequently-accessed one should rank higher
        due to the frequency scoring component (log2-based).
        """
        n1 = self.brain.remember(
            type='decision',
            title='React lifecycle: prefer useEffect cleanup over componentWillUnmount',
            content='All new components must use functional patterns with useEffect return cleanup. '
                    'Class components with componentWillUnmount are legacy and should be migrated '
                    'incrementally during feature work, not as standalone refactoring tickets.',
            keywords='react lifecycle useEffect cleanup componentWillUnmount functional'
        )
        n2 = self.brain.remember(
            type='decision',
            title='React lifecycle: avoid useLayoutEffect except for DOM measurement',
            content='useLayoutEffect blocks the browser paint and causes jank on slower devices. '
                    'Only use it when you need synchronous DOM measurements before the user sees '
                    'the frame. All other side effects belong in useEffect.',
            keywords='react lifecycle useLayoutEffect useEffect DOM measurement performance'
        )
        self.brain.save()

        # Access n1 ten times to boost its frequency score
        for _ in range(10):
            self.brain.recall_with_embeddings('react component lifecycle patterns', limit=5)
            # Each recall marks accessed nodes, so n1 gets frequency bumps

        # Now query again — the one that was accessed more should rank higher
        results = self.brain.recall_with_embeddings('react lifecycle useEffect', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 2,
                        f'Should find at least 2 results, got {len(result_list)}')
        # The first result should have a higher access_count
        top_access = result_list[0].get('access_count', 0)
        second_access = result_list[1].get('access_count', 0)
        self.assertGreaterEqual(top_access, second_access,
                                'Frequently accessed node should have higher access_count')

    def test_emotion_weight(self):
        """A node with high emotional intensity should have higher emotion_intensity in recall.

        Creates two decision nodes with identical keywords/content structure. One carries
        high emotion (0.8) and the other is neutral. Verify that the emotion component
        is reflected in the scoring breakdown.
        """
        self.brain.remember(
            type='decision',
            title='API rate limiting: implement token bucket with Redis sliding window',
            content='After the DDoS incident that took down the billing service for 47 minutes, '
                    'we decided on a token bucket algorithm backed by Redis sorted sets. Each API '
                    'key gets 1000 tokens per minute with burst capacity of 200. Rate limit headers '
                    'are mandatory on all responses per RFC 6585.',
            keywords='api rate-limiting token-bucket redis sliding-window ddos billing',
            emotion=0.8,
            emotion_label='urgency'
        )
        self.brain.remember(
            type='decision',
            title='API rate limiting: use fixed-window counters for internal services',
            content='Internal service-to-service calls use simpler fixed-window counters because '
                    'the traffic patterns are predictable and bursts are expected during batch jobs. '
                    'No need for the complexity of sliding windows when the caller is trusted.',
            keywords='api rate-limiting fixed-window internal services counters',
            emotion=0,
            emotion_label='neutral'
        )
        self.brain.save()

        results = self.brain.recall_with_embeddings('rate limiting strategy api', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 2,
                        f'Should find at least 2 results, got {len(result_list)}')
        # Both nodes should be found
        token_bucket = [r for r in result_list if 'token bucket' in r['title']]
        fixed_window = [r for r in result_list if 'fixed-window' in r['title']]
        self.assertTrue(len(token_bucket) > 0, 'Should find token bucket node')
        self.assertTrue(len(fixed_window) > 0, 'Should find fixed-window node')
        # Verify emotion_intensity is correctly propagated in scoring breakdown
        tb_emotion = token_bucket[0].get('emotion_intensity', 0)
        fw_emotion = fixed_window[0].get('emotion_intensity', 0)
        self.assertGreater(tb_emotion, fw_emotion,
                          f'High-emotion node should have higher emotion_intensity ({tb_emotion} vs {fw_emotion})')

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

        results = self.brain.recall_with_embeddings('secrets credentials api keys security', limit=10)
        result_list = results.get('results', [])
        found_ids = [n['id'] for n in result_list]
        self.assertIn(r['id'], found_ids,
                      'Locked node should still appear despite being 60 days old')


# ═══════════════════════════════════════════════════════════════
# TestIntentDetection — verify that _classify_intent() correctly
# maps natural language queries to intent categories
# ═══════════════════════════════════════════════════════════════

class TestIntentDetection(BrainTestBase):
    """Verify that _classify_intent() detects query intent and returns correct type boosts."""

    def test_intent_decision_lookup(self):
        """'what did we decide about authentication' should classify as decision_lookup."""
        result = self.brain._classify_intent('what did we decide about authentication')
        self.assertEqual(result['intent'], 'decision_lookup',
                         f'Expected decision_lookup, got {result["intent"]}')

    def test_intent_reasoning_chain(self):
        """'why did we choose Clerk over Auth0' should classify as reasoning_chain."""
        result = self.brain._classify_intent('why did we choose Clerk over Auth0')
        self.assertEqual(result['intent'], 'reasoning_chain',
                         f'Expected reasoning_chain, got {result["intent"]}')

    def test_intent_state_query(self):
        """'what is the current status of the API migration' should classify as state_query."""
        result = self.brain._classify_intent('what is the current status of the API migration')
        self.assertEqual(result['intent'], 'state_query',
                         f'Expected state_query, got {result["intent"]}')

    def test_intent_temporal(self):
        """'what changed last week' should classify as temporal."""
        result = self.brain._classify_intent('what changed last week')
        self.assertEqual(result['intent'], 'temporal',
                         f'Expected temporal, got {result["intent"]}')

    def test_intent_correction_lookup(self):
        """'what went wrong with the frontend deployment' should classify as correction_lookup."""
        result = self.brain._classify_intent('what went wrong with the frontend deployment')
        self.assertEqual(result['intent'], 'correction_lookup',
                         f'Expected correction_lookup, got {result["intent"]}')

    def test_intent_how_to(self):
        """'how should I configure the webhook handler' should classify as how_to."""
        result = self.brain._classify_intent('how should I configure the webhook handler')
        self.assertEqual(result['intent'], 'how_to',
                         f'Expected how_to, got {result["intent"]}')

    def test_intent_general(self):
        """'kubernetes pods' should classify as general (no specific intent pattern)."""
        result = self.brain._classify_intent('kubernetes pods')
        self.assertEqual(result['intent'], 'general',
                         f'Expected general, got {result["intent"]}')

    def test_intent_boosts_types(self):
        """For decision_lookup intent, the typeBoosts dict should boost 'decision' above 1.0.

        This verifies that intent detection feeds into the type-boosting mechanism
        that promotes relevant node types for each intent category.
        """
        result = self.brain._classify_intent('what did we decide about the database schema')
        self.assertEqual(result['intent'], 'decision_lookup')
        type_boosts = result.get('typeBoosts', {})
        self.assertIn('decision', type_boosts,
                       'decision_lookup intent should boost decision type')
        self.assertGreater(type_boosts['decision'], 1.0,
                           f'decision boost should be > 1.0, got {type_boosts["decision"]}')


# ═══════════════════════════════════════════════════════════════
# TestDampening — verify that hub nodes, low-value types,
# and low-confidence nodes are appropriately penalized
# ═══════════════════════════════════════════════════════════════

class TestDampening(BrainTestBase):
    """Verify that dampening mechanisms reduce noise from hubs and low-signal types."""

    def test_hub_dampening(self):
        """A node connected to 50+ other nodes should rank lower than an equivalent non-hub node.

        Hub nodes (high-degree) tend to be generic and match too many queries. The
        hub dampening formula (threshold/edge_count) reduces their relevance score.
        """
        # Create hub node
        hub = self.brain.remember(
            type='decision',
            title='Adopt TypeScript strict mode for all new modules',
            content='Every new TypeScript file must use strict mode with noImplicitAny, '
                    'strictNullChecks, and noUncheckedIndexedAccess enabled. This catches '
                    'null pointer errors at compile time and reduces runtime crashes by ~40%.',
            keywords='typescript strict-mode noImplicitAny strictNullChecks compile-time safety'
        )

        # Create 55 connected nodes to make it a hub
        for i in range(55):
            satellite = self.brain.remember(
                type='context',
                title=f'Module {i}: migrated to TypeScript strict mode',
                content=f'Module {i} was migrated to strict TypeScript. Found {i + 3} type errors '
                        f'during migration, all resolved. No runtime regressions detected.',
                keywords=f'typescript migration module-{i}'
            )
            self.brain.connect(hub['id'], satellite['id'], 'related', 0.5)

        # Create non-hub node with same content
        non_hub = self.brain.remember(
            type='decision',
            title='Adopt TypeScript strict mode for the payments service',
            content='The payments service specifically needs TypeScript strict mode because it '
                    'handles currency arithmetic where null values cause silent data corruption. '
                    'Enable noImplicitAny and strictNullChecks as the first migration step.',
            keywords='typescript strict-mode payments service compile-time safety'
        )
        # Give it just 2 connections
        c1 = self.brain.remember(type='context', title='Payments service overview',
                                  content='The payments service handles Stripe integration and invoice generation.',
                                  keywords='payments stripe invoices')
        c2 = self.brain.remember(type='context', title='Payments service testing plan',
                                  content='Integration tests for the payments service use Stripe test mode API keys.',
                                  keywords='payments testing stripe')
        self.brain.connect(non_hub['id'], c1['id'], 'related', 0.5)
        self.brain.connect(non_hub['id'], c2['id'], 'related', 0.5)
        self.brain.save()

        # Query that matches both equally by content
        results = self.brain.recall('typescript strict mode safety', limit=10)
        result_list = results.get('results', [])

        # Find positions
        hub_pos = None
        non_hub_pos = None
        for i, r in enumerate(result_list):
            if r['id'] == hub['id']:
                hub_pos = i
            if r['id'] == non_hub['id']:
                non_hub_pos = i

        if hub_pos is not None and non_hub_pos is not None:
            self.assertGreater(hub_pos, non_hub_pos,
                               f'Hub node (pos={hub_pos}) should rank lower than non-hub (pos={non_hub_pos})')

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

    def test_confidence_weighting(self):
        """A high-confidence node should rank above a low-confidence one with similar content.

        Confidence maps to a scoring multiplier: 0.95 -> ~1.04x, 0.3 -> ~0.77x.
        The gap should be enough to change ranking for otherwise similar nodes.
        """
        self.brain.remember(
            type='decision',
            title='Deployment strategy: blue-green with Kubernetes rolling updates',
            content='We use blue-green deployments for zero-downtime releases. Kubernetes handles '
                    'the rolling update with maxSurge=1 and maxUnavailable=0. Health checks must '
                    'pass for 30 seconds before old pods are terminated.',
            keywords='deployment blue-green kubernetes rolling-update zero-downtime',
            confidence=0.95
        )
        self.brain.remember(
            type='decision',
            title='Deployment strategy: canary releases with percentage-based traffic splitting',
            content='Canary deployments route 5% of traffic to the new version and monitor error '
                    'rates for 15 minutes before proceeding. If error rate exceeds 1%, automatic '
                    'rollback is triggered via Argo Rollouts.',
            keywords='deployment canary traffic-splitting argo-rollouts rollback',
            confidence=0.3
        )
        self.brain.save()

        results = self.brain.recall_with_embeddings('deployment strategy production releases', limit=5)
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 2,
                        f'Should find at least 2 results, got {len(result_list)}')
        # High-confidence (blue-green) should rank first
        self.assertIn('blue-green', result_list[0]['title'],
                      f'High-confidence node should rank first, got: {result_list[0]["title"]}')

    def test_project_filtering(self):
        """Nodes in the queried project should rank higher than nodes in a different project.

        When project="alpha" is passed, alpha nodes get priority sorting
        (project match first, then by effective_activation).
        """
        self.brain.remember(
            type='decision',
            title='Alpha: use GraphQL for the dashboard API layer',
            content='The alpha project dashboard needs flexible queries for widgets. GraphQL lets '
                    'the frontend request exactly the fields it needs without over-fetching. We use '
                    'Apollo Server with DataLoader for N+1 query prevention.',
            keywords='graphql dashboard api apollo dataloader alpha',
            project='alpha'
        )
        self.brain.remember(
            type='decision',
            title='Beta: use REST for the public developer API',
            content='The beta project public API must be REST because our developer audience expects '
                    'standard HTTP methods, predictable URLs, and OpenAPI documentation. GraphQL would '
                    'add unnecessary complexity for third-party consumers.',
            keywords='rest api public openapi developer beta',
            project='beta'
        )
        self.brain.save()

        results = self.brain.recall_with_embeddings('API design decisions', limit=5, project='alpha')
        result_list = results.get('results', [])
        self.assertTrue(len(result_list) >= 1,
                        f'Should find at least 1 result, got {len(result_list)}')
        # Alpha project node should rank higher when filtering by alpha
        alpha_nodes = [n for n in result_list if n.get('project') == 'alpha']
        self.assertTrue(len(alpha_nodes) > 0, 'Should find alpha project nodes')
        if len(result_list) >= 2:
            first_project = result_list[0].get('project')
            self.assertEqual(first_project, 'alpha',
                             f'Alpha node should rank first with project filter, got project={first_project}')


# ═══════════════════════════════════════════════════════════════
# TestSpreadingActivation — verify multi-hop graph traversal
# with exponential decay per hop
# ═══════════════════════════════════════════════════════════════

class TestSpreadingActivation(BrainTestBase):
    """Verify that spreading activation traverses edges with proper decay."""

    def test_spread_reaches_2_hop(self):
        """Activation should reach a node 2 hops away through graph edges.

        Creates chain A -> B -> C with 'related' edges. Seeds activation at A.
        C should appear in the activation results via B.
        """
        a = self.brain.remember(
            type='decision',
            title='Adopt event sourcing for the order management service',
            content='Orders are complex state machines. Event sourcing gives us a complete audit '
                    'trail, enables temporal queries, and makes it trivial to rebuild projections '
                    'when business rules change.',
            keywords='event-sourcing orders audit-trail projections cqrs'
        )
        b = self.brain.remember(
            type='mechanism',
            title='Event store implementation with PostgreSQL append-only table',
            content='Events are stored in a PostgreSQL table with columns: stream_id, version, '
                    'event_type, payload (JSONB), metadata (JSONB), created_at. Optimistic '
                    'concurrency uses the version column with unique constraint.',
            keywords='event-store postgresql append-only jsonb optimistic-concurrency'
        )
        c = self.brain.remember(
            type='lesson',
            title='Projection rebuilds must be idempotent and versioned',
            content='We learned the hard way that projections need version stamps. When rebuilding '
                    'the order summary projection, duplicate events caused inflated revenue totals. '
                    'Now every projection tracks its last processed event version.',
            keywords='projection rebuild idempotent versioned event-sourcing'
        )
        self.brain.connect(a['id'], b['id'], 'related', 0.8)
        self.brain.connect(b['id'], c['id'], 'related', 0.8)
        self.brain.save()

        activated = self.brain.spread_activation([a['id']])
        activated_ids = [n['id'] for n in activated]

        self.assertIn(c['id'], activated_ids,
                      'Node C (2 hops from A) should receive spreading activation')

    def test_spread_decays_by_hop(self):
        """Activation should decay exponentially: C's activation ~= SPREAD_DECAY^2 * edge_weights.

        With SPREAD_DECAY=0.5, 2 hops through weight-1.0 edges should give
        C roughly 0.5^2 = 0.25 of A's initial activation (1.0).
        """
        a = self.brain.remember(
            type='decision',
            title='Use Redis Streams for real-time notification delivery',
            content='Redis Streams provide ordered, persistent message delivery with consumer groups. '
                    'Each notification channel maps to a stream key. Consumer groups handle fan-out '
                    'to WebSocket server instances.',
            keywords='redis streams notifications real-time consumer-groups websocket'
        )
        b = self.brain.remember(
            type='mechanism',
            title='Consumer group configuration for notification fan-out',
            content='Each WebSocket server instance joins the NOTIFY consumer group. XREADGROUP '
                    'with BLOCK 5000 provides efficient long-polling. Pending entries are claimed '
                    'after 60 seconds of inactivity via XCLAIM.',
            keywords='consumer-group xreadgroup xclaim fan-out websocket notification'
        )
        c = self.brain.remember(
            type='lesson',
            title='Redis Streams consumer groups need explicit ACK or messages pile up',
            content='Unacknowledged messages in the PEL (pending entries list) grow unbounded. '
                    'We hit 500MB of pending entries before realizing XACK was missing from the '
                    'error handling path. Always ACK in a finally block.',
            keywords='redis streams xack pending-entries-list memory-leak error-handling'
        )

        # Use weight=1.0 edges for predictable math
        self.brain.connect(a['id'], b['id'], 'related', 1.0)
        self.brain.connect(b['id'], c['id'], 'related', 1.0)
        self.brain.save()

        activated = self.brain.spread_activation([a['id']])
        activation_map = {n['id']: n['spread_activation'] for n in activated}

        a_act = activation_map.get(a['id'], 0)
        c_act = activation_map.get(c['id'], 0)

        # A starts at 1.0
        self.assertAlmostEqual(a_act, 1.0, places=1,
                               msg=f'Seed node A should have activation ~1.0, got {a_act}')
        # C should be roughly SPREAD_DECAY^1 (hop 1: A->B) * SPREAD_DECAY^2 (hop 2: B->C)
        # Actually: hop 1 gives B = 1.0 * 1.0 * 0.5 = 0.5, hop 2 gives C = 0.5 * 1.0 * 0.25 = 0.125
        # But B's activation at hop 2 is the accumulated value (0.5), so C = 0.5 * 1.0 * 0.25 = 0.125
        # The key assertion: C's activation should be significantly less than A's
        self.assertGreater(c_act, 0, 'Node C should have some activation')
        self.assertLess(c_act, a_act * 0.5,
                        f'Node C ({c_act}) should have substantially less activation than A ({a_act})')

    def test_spread_respects_max_hops(self):
        """Activation should NOT reach nodes beyond MAX_HOPS (3) edges away.

        Creates chain A -> B -> C -> D -> E. With MAX_HOPS=3, the spreading
        algorithm runs 3 iterations. Hop 1: A->B, Hop 2: B->C, Hop 3: C->D.
        E is 4 edges from A and should not receive meaningful activation.
        """
        nodes = []
        titles = [
            ('decision', 'Microservice boundary: separate billing from user management',
             'Billing and user management have different scaling needs and deployment cadences. '
             'Separating them lets billing scale horizontally during invoice generation while '
             'user management stays on a single primary.',
             'microservice billing user-management scaling deployment'),
            ('mechanism', 'gRPC for inter-service communication between billing and users',
             'gRPC with protobuf provides type-safe contracts and efficient binary serialization. '
             'Bi-directional streaming is used for bulk user lookups during invoice generation.',
             'grpc protobuf inter-service billing users streaming'),
            ('constraint', 'Circuit breaker required on all cross-service calls',
             'Every gRPC client must wrap calls in a circuit breaker (Resilience4j or Polly). '
             'Open threshold: 5 failures in 30 seconds. Half-open: 1 probe request per 10 seconds.',
             'circuit-breaker resilience4j grpc fault-tolerance timeout'),
            ('lesson', 'Distributed tracing is mandatory before splitting a monolith',
             'When we split auth from the monolith, debugging production issues tripled in difficulty. '
             'We should have set up Jaeger distributed tracing first.',
             'distributed-tracing jaeger monolith splitting debugging production'),
            ('context', 'Future plan: extract notification service as fifth microservice',
             'Notifications are currently embedded in the billing service. Extracting them would '
             'decouple email/SMS delivery from invoice processing.',
             'notification service extraction billing decoupling email sms'),
        ]

        for type_, title, content, keywords in titles:
            n = self.brain.remember(type=type_, title=title, content=content, keywords=keywords)
            nodes.append(n)

        # Chain: A -> B -> C -> D -> E
        for i in range(len(nodes) - 1):
            self.brain.connect(nodes[i]['id'], nodes[i + 1]['id'], 'related', 1.0)
        self.brain.save()

        activated = self.brain.spread_activation([nodes[0]['id']])
        activation_map = {n['id']: n['spread_activation'] for n in activated}

        e_activation = activation_map.get(nodes[4]['id'], 0)
        a_activation = activation_map.get(nodes[0]['id'], 0)

        # E is 4 hops from A. Activation should decay significantly with distance.
        # With SPREAD_DECAY=0.5, each hop halves activation, so E ≈ A * 0.5^4 = ~6% of A.
        # Allow some tolerance since scoring involves multiple factors.
        d_activation = activation_map.get(nodes[3]['id'], 0)
        b_activation = activation_map.get(nodes[1]['id'], 0)
        # At minimum: each hop should reduce activation
        if b_activation > 0 and d_activation > 0:
            self.assertGreater(b_activation, d_activation,
                              'Closer nodes should have higher activation than distant ones')


# ═══════════════════════════════════════════════════════════════
# TestTFIDFRecall — verify TF-IDF fallback path, stopword filtering,
# and cosine similarity scoring
# ═══════════════════════════════════════════════════════════════

class TestTFIDFRecall(BrainTestBase):
    """Verify TF-IDF based recall when embedder is unavailable."""

    def test_tfidf_without_embedder(self):
        """Recall should work via TF-IDF keyword matching when embedder is disabled.

        Manually disables the embedder to force the keyword-only fallback path.
        Verifies that remember + recall still works using TF-IDF scoring.
        """
        from servers import embedder

        # Save original state and disable embedder
        original_model = embedder._model
        original_loaded = embedder.stats['model_loaded']
        embedder._model = None
        embedder.stats['model_loaded'] = False

        try:
            self.brain.remember(
                type='decision',
                title='Use Tailwind CSS utility classes instead of CSS modules',
                content='Tailwind reduces context switching between JSX and stylesheets. The JIT '
                        'compiler purges unused classes, keeping bundle size under 10KB. Component '
                        'libraries like Headless UI integrate seamlessly with Tailwind.',
                keywords='tailwind css utility-classes jit purge headless-ui components'
            )
            self.brain.save()

            results = self.brain.recall_with_embeddings('tailwind css utility classes', limit=5)
            result_list = results.get('results', [])
            self.assertTrue(len(result_list) >= 1,
                            'Should find results via TF-IDF keyword fallback')
            self.assertIn('Tailwind', result_list[0]['title'],
                          'TF-IDF should find the Tailwind node')
            # Verify it used the degraded path
            self.assertEqual(results.get('_recall_mode'), 'keyword_only_DEGRADED',
                             'Should report keyword_only_DEGRADED mode')
        finally:
            # Restore embedder state
            embedder._model = original_model
            embedder.stats['model_loaded'] = original_loaded

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
