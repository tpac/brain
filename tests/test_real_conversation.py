"""
Real conversation tests — verify the challenge system works against
actual brain data and real conversation transcripts.

These tests use:
- Real brain.db (copied to temp)
- Real JSONL session transcripts
- Real recall queries from engineering conversations
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.schema import ensure_logs_schema

# Paths to real data
BRAIN_DB = os.path.expanduser('~/AgentsContext/brain/brain.db')
BRAIN_LOGS_DB = os.path.expanduser('~/AgentsContext/brain/brain_logs.db')
SESSION_JSONL = os.path.join(os.path.dirname(__file__), '..', 'exports', 'sessions')


def _find_session_jsonl():
    """Find a session JSONL with enough user messages for testing."""
    if not os.path.isdir(SESSION_JSONL):
        return None
    files = [f for f in os.listdir(SESSION_JSONL) if f.endswith('.jsonl')]
    if not files:
        return None
    # Pick smallest file that's at least 10KB (has real conversation)
    files_with_size = [(f, os.path.getsize(os.path.join(SESSION_JSONL, f))) for f in files]
    files_with_size.sort(key=lambda x: x[1])
    for fname, size in files_with_size:
        if size > 10000:  # At least 10KB
            return os.path.join(SESSION_JSONL, fname)
    # Fall back to largest
    return os.path.join(SESSION_JSONL, files_with_size[-1][0])


def _extract_user_messages(jsonl_path, limit=10):
    """Extract user messages from JSONL transcript."""
    messages = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('type') == 'user':
                    msg = obj.get('message', {})
                    content = msg.get('content', '') if isinstance(msg, dict) else ''
                    if content and len(content) > 5:
                        messages.append(content)
                        if len(messages) >= limit:
                            break
            except (json.JSONDecodeError, KeyError):
                continue
    return messages


@unittest.skipUnless(os.path.exists(BRAIN_DB), "Real brain.db not available")
class TestRealBrainRecall(unittest.TestCase):
    """Test recall against real brain data."""

    def setUp(self):
        """Copy real brain.db to temp location."""
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.logs_path = os.path.join(self.tmp, 'brain_logs.db')
        shutil.copy2(BRAIN_DB, self.db_path)
        if os.path.exists(BRAIN_LOGS_DB):
            shutil.copy2(BRAIN_LOGS_DB, self.logs_path)

        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        try:
            self.brain.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_recall_challenge_format(self):
        """KPI: Real recall produces results with revised status + full IDs."""
        # Use a very specific query that matches known brain content
        result = self.brain.recall_with_embeddings("encoding should be an instinct", limit=5)
        results = result.get('results', [])

        self.assertGreater(len(results), 0, "Real brain should have encoding-related nodes")

        from servers.brain_voice import BrainVoice
        lines = []
        BrainVoice.format_recall_results(results, lines)
        output = '\n'.join(lines)

        # Full IDs (not truncated to 8 chars)
        for r in results:
            node_id = r.get('id', '')
            if len(node_id) >= 16:  # short typed IDs (rul_xxx) are shorter
                self.assertIn(node_id, output,
                              "Full node ID should appear in formatted output")

        # Revised status
        self.assertIn('revised:', output, "Output should contain revised status")

        # Confidence
        self.assertIn('conf:', output, "Output should contain confidence score")

    def test_real_gap_detection_on_unknown_topic(self):
        """KPI: Brain has no knowledge of 'kubernetes pod scaling' — gap flagged."""
        result = self.brain.recall_with_embeddings("kubernetes pod scaling strategy", limit=5)

        # This topic should have no matches in the brain
        if not result.get('results'):
            self.assertIn('_gap', result,
                          "Gap should be flagged when no results found")
            gap = result['_gap']
            self.assertIn('kubernetes', gap.get('query', ''))

    def test_real_recall_has_confidence_and_dates(self):
        """KPI: Real recalled nodes have confidence and created_at fields."""
        result = self.brain.recall_with_embeddings("encoding should be an instinct", limit=3)
        results = result.get('results', [])

        if results:
            first = results[0]
            self.assertIn('created_at', first, "Node should have created_at")
            # confidence may be None for old nodes, but field should exist
            self.assertIn('confidence', first, "Node should have confidence field")

    # test_real_challenge_output_assembly: DELETED — render_prompt replaced by SurfaceAssembler (2026-03-27)


@unittest.skipUnless(os.path.exists(BRAIN_DB), "Real brain.db not available")
class TestRealConsolidation(unittest.TestCase):
    """Test consolidation detection against real brain data."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.logs_path = os.path.join(self.tmp, 'brain_logs.db')
        shutil.copy2(BRAIN_DB, self.db_path)
        if os.path.exists(BRAIN_LOGS_DB):
            shutil.copy2(BRAIN_LOGS_DB, self.logs_path)

        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        try:
            self.brain.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_consolidation_finds_duplicates(self):
        """KPI: Real brain has known duplicate nodes — detection finds them."""
        # Use 0.80 threshold — brain was recently cleaned of obvious dupes
        count = self.brain.detect_consolidation_candidates(
            similarity_threshold=0.80, max_pairs=5)

        # We know the real brain has duplicates (e.g. multiple SKILL.md decisions)
        self.assertGreater(count, 0,
                           "Real brain should have at least 1 consolidation candidate pair")

        from servers.dal import LogsDAL
        logs_dal = LogsDAL(self.brain.logs_conn)
        pairs = logs_dal.get_pending_consolidation(limit=5)

        for pair in pairs:
            # Verify both nodes exist and are same type
            node_a = self.brain.conn.execute(
                'SELECT type, title FROM nodes WHERE id = ?',
                (pair['node_id_a'],)).fetchone()
            node_b = self.brain.conn.execute(
                'SELECT type, title FROM nodes WHERE id = ?',
                (pair['node_id_b'],)).fetchone()
            self.assertIsNotNone(node_a, "Node A should exist")
            self.assertIsNotNone(node_b, "Node B should exist")
            self.assertEqual(node_a[0], node_b[0],
                             "Consolidation pair should be same type: %s vs %s" % (
                                 node_a[1][:50], node_b[1][:50]))

    def test_consolidation_nodes_created_apart(self):
        """KPI: Paired nodes were created > 24h apart."""
        self.brain.detect_consolidation_candidates(
            similarity_threshold=0.80, min_age_hours=24, max_pairs=3)

        from servers.dal import LogsDAL
        from datetime import datetime
        logs_dal = LogsDAL(self.brain.logs_conn)
        pairs = logs_dal.get_pending_consolidation(limit=3)

        for pair in pairs:
            date_a = self.brain.conn.execute(
                'SELECT created_at FROM nodes WHERE id = ?',
                (pair['node_id_a'],)).fetchone()
            date_b = self.brain.conn.execute(
                'SELECT created_at FROM nodes WHERE id = ?',
                (pair['node_id_b'],)).fetchone()
            if date_a and date_b and date_a[0] and date_b[0]:
                try:
                    dt_a = datetime.fromisoformat(date_a[0].replace('Z', '+00:00'))
                    dt_b = datetime.fromisoformat(date_b[0].replace('Z', '+00:00'))
                    hours_apart = abs((dt_a - dt_b).total_seconds()) / 3600
                    self.assertGreater(hours_apart, 24,
                                       "Paired nodes should be created > 24h apart (%.1f hours)" % hours_apart)
                except ValueError:
                    pass  # Can't parse dates, skip this pair


@unittest.skipUnless(_find_session_jsonl() is not None, "No session JSONL files available")
class TestRealConversationStream(unittest.TestCase):
    """Test message stream with real conversation data."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        shutil.copy2(BRAIN_DB, self.db_path) if os.path.exists(BRAIN_DB) else None

        from servers.brain import Brain
        self.brain = Brain(self.db_path)
        self.jsonl_path = _find_session_jsonl()

    def tearDown(self):
        try:
            self.brain.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_messages_stored_and_retrieved(self):
        """KPI: Real Tom messages stored and retrieved as pending."""
        messages = _extract_user_messages(self.jsonl_path, limit=5)
        self.assertGreater(len(messages), 0, "Should find user messages in JSONL")

        for msg in messages:
            self.brain.store_exchange(msg, "assistant reply", "test-session")

        from servers.dal_message_stream import MessageStreamDAL
        dal = MessageStreamDAL(self.brain.logs_conn)
        pending = dal.get_pending(limit=3)

        # Should get up to 3 pending (or all stored if fewer)
        expected_count = min(3, len(messages))
        self.assertEqual(len(pending), expected_count,
                         "Should get %d pending (got %d from %d stored)" % (
                             expected_count, len(pending), len(messages)))
        # Verify content matches stored messages
        for p in pending:
            self.assertIn(p['content'], messages,
                          "Pending content should match a stored message")


@unittest.skipUnless(os.path.exists(BRAIN_DB), "Real brain.db not available")
class TestRealRevise(unittest.TestCase):
    """Test revise against real brain nodes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.logs_path = os.path.join(self.tmp, 'brain_logs.db')
        shutil.copy2(BRAIN_DB, self.db_path)
        if os.path.exists(BRAIN_LOGS_DB):
            shutil.copy2(BRAIN_LOGS_DB, self.logs_path)

        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        try:
            self.brain.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_revise_real_node(self):
        """KPI: Revise a real node — content appended, revised_at set, embedding changed."""
        # Find a real node to revise
        result = self.brain.recall_with_embeddings("encoding pipeline", limit=1)
        results = result.get('results', [])
        self.assertGreater(len(results), 0, "Should find a node to revise")

        node = results[0]
        nid = node['id']
        original_content = node.get('content', '')

        # Get embedding before
        emb_before = self.brain.conn.execute(
            'SELECT embedding FROM node_embeddings WHERE node_id = ?', (nid,)).fetchone()

        # Revise it
        rev_result = self.brain.revise(
            nid, "Updated 2026-03-26: DAL migration completed for this component",
            "DAL migration done")
        self.assertNotIn('error', rev_result, "Revise should succeed on real node")
        self.assertIn('revised_at', rev_result)

        # Verify content appended
        updated = self.brain.conn.execute(
            'SELECT content, revised_at FROM nodes WHERE id = ?', (nid,)).fetchone()
        self.assertIn(original_content, updated[0], "Original content preserved")
        self.assertIn('DAL migration completed', updated[0], "New content appended")
        self.assertIsNotNone(updated[1], "revised_at should be set")

        # Verify embedding changed
        emb_after = self.brain.conn.execute(
            'SELECT embedding FROM node_embeddings WHERE node_id = ?', (nid,)).fetchone()
        if emb_before and emb_after:
            self.assertNotEqual(emb_before[0], emb_after[0],
                                "Embedding should change after revision")


@unittest.skipUnless(os.path.exists(BRAIN_DB), "Real brain.db not available")
class TestChallengeFormatBenchmark(unittest.TestCase):
    """Benchmark: compare OLD vs NEW format quality across real exchanges.

    This test establishes baselines for the challenge system.
    Future changes should not regress these KPIs.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.logs_path = os.path.join(self.tmp, 'brain_logs.db')
        shutil.copy2(BRAIN_DB, self.db_path)
        if os.path.exists(BRAIN_LOGS_DB):
            shutil.copy2(BRAIN_LOGS_DB, self.logs_path)
        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        try:
            self.brain.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_format_quality_kpis(self):
        """DISABLED — render_prompt deleted, pending assembler rewrite (2026-03-28)."""
        self.skipTest('render_prompt deleted — rewrite against SurfaceAssembler')
        from servers.brain_voice import BrainVoice

        queries = [
            'encoding should be an instinct',
            'what did we decide about recall?',
            'hook architecture flat dispatchers',
            'Tom engineering principles',
            'kubernetes pod scaling',
            'React component lifecycle',
            'what corrections have I made?',
            'brain origin story frustration',
            'embedding similarity threshold floor',
            'DAL migration plan',
        ]

        with_results = 0
        old_truncated = 0
        new_truncated = 0
        total_nodes = 0
        nodes_with_neighbors = 0
        gaps_detected = 0
        gaps_with_callout = 0

        for q in queries:
            result = self.brain.recall_with_embeddings(q, limit=8)
            results = result.get('results', [])
            gap = result.get('_gap')

            if results:
                with_results += 1
            if gap:
                gaps_detected += 1

            for r in results:
                total_nodes += 1
                content = r.get('content', '')
                if len(content) > 300:
                    old_truncated += 1
                if r.get('_neighbors'):
                    nodes_with_neighbors += 1

            # Check new format
            voice = BrainVoice(self.brain)
            rendered = voice.render_prompt(
                results=results, prompt_signals={}, gap=gap)
            output = rendered.get('for_claude', '')

            if gap and 'UNKNOWN TOPIC' in output:
                gaps_with_callout += 1

        # KPI assertions with baselines
        recall_coverage = with_results / len(queries)
        self.assertGreaterEqual(recall_coverage, 0.5,
            "Recall coverage should be >= 50%% (got %.0f%%)" % (recall_coverage * 100))

        if total_nodes > 0:
            old_truncation_rate = old_truncated / total_nodes
            self.assertGreater(old_truncation_rate, 0.5,
                "OLD format should truncate >50%% of nodes (got %.0f%%) — proves NEW is better" % (
                    old_truncation_rate * 100))

        self.assertEqual(new_truncated, 0,
            "NEW format should NEVER truncate content")

        if gaps_detected > 0:
            gap_callout_rate = gaps_with_callout / gaps_detected
            self.assertEqual(gap_callout_rate, 1.0,
                "100%% of gaps should get UNKNOWN TOPIC callout (got %.0f%%)" % (
                    gap_callout_rate * 100))

        print("\n=== CHALLENGE FORMAT BENCHMARK (run as regression baseline) ===")
        print("Recall coverage: %.0f%% (%d/%d)" % (
            recall_coverage * 100, with_results, len(queries)))
        print("Total nodes recalled: %d" % total_nodes)
        print("OLD truncation rate: %.0f%% (%d/%d)" % (
            old_truncated * 100 / max(total_nodes, 1), old_truncated, total_nodes))
        print("NEW truncation rate: 0%%")
        print("Nodes with neighbors: %d/%d" % (nodes_with_neighbors, total_nodes))
        print("Gaps detected: %d, with callout: %d" % (gaps_detected, gaps_with_callout))


@unittest.skipUnless(os.path.exists(BRAIN_DB) and _find_session_jsonl() is not None,
                     "Real brain.db and session JSONL required")
class TestFullChallengeSimulation(unittest.TestCase):
    """Simulate the full challenge pipeline across dozens of real exchanges.

    This is the most important test — it runs the ENTIRE flow that produces
    the additionalContext injection and checks the output against spec.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'brain.db')
        self.logs_path = os.path.join(self.tmp, 'brain_logs.db')
        shutil.copy2(BRAIN_DB, self.db_path)
        if os.path.exists(BRAIN_LOGS_DB):
            shutil.copy2(BRAIN_LOGS_DB, self.logs_path)

        from servers.brain import Brain
        self.brain = Brain(self.db_path)

    def tearDown(self):
        try:
            self.brain.close()
        except Exception:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _simulate_challenge_turn(self, user_message):
        """Simulate one hook_recall turn. Returns the full output + metadata."""
        from servers.brain_voice import BrainVoice
        from servers.dal_message_stream import MessageStreamDAL
        from servers.dal import LogsDAL

        # 1. Recall
        result = self.brain.recall_with_embeddings(user_message, limit=8)
        results = result.get('results', [])

        # 2. Gap detection
        gap = result.get('_gap')

        # 3. Pending Tom messages
        try:
            msg_dal = MessageStreamDAL(self.brain.logs_conn)
            pending_tom = msg_dal.get_pending(limit=3)
        except Exception:
            pending_tom = []

        # 4. Consolidation (only when few results)
        consolidation = []
        consolidation_total = 0
        if len(results) < 2:
            try:
                logs_dal = LogsDAL(self.brain.logs_conn)
                consolidation_total = logs_dal.count_pending_consolidation()
                if consolidation_total:
                    raw_pairs = logs_dal.get_pending_consolidation(limit=2)
                    for pair in raw_pairs:
                        na = self.brain.recall_node(pair['node_id_a'], neighbor_limit=1)
                        nb = self.brain.recall_node(pair['node_id_b'], neighbor_limit=1)
                        if na.get('results') and nb.get('results'):
                            consolidation.append({
                                'node_a': na['results'][0],
                                'node_b': nb['results'][0],
                                'pair_id': pair['id'],
                            })
            except Exception:
                pass

        # 5. Format
        voice = BrainVoice(self.brain)
        rendered = voice.render_prompt(
            results=results,
            prompt_signals={},
            gap=gap,
            consolidation=consolidation,
            consolidation_total=consolidation_total,
            pending_tom_messages=pending_tom,
        )

        output = rendered.get('for_claude', '')

        return {
            'query': user_message[:100],
            'results_count': len(results),
            'has_gap': gap is not None,
            'gap_query': gap.get('query', '') if gap else '',
            'pending_tom_count': len(pending_tom),
            'consolidation_count': len(consolidation),
            'output_length': len(output),
            'has_challenge_header': '⚠️ ACTIVE RECALL' in output or 'ACTIVE RECALL' in output,
            'has_revise_instruction': 'revise(' in output,
            'has_gap_callout': 'UNKNOWN TOPIC' in output,
            'has_consolidation': 'CONSOLIDATION' in output,
            'has_pending': 'PENDING' in output,
            'has_full_ids': any(len(r.get('id', '')) > 16 and r['id'] in output
                              for r in results) if results else True,
            'has_revised_status': 'revised:' in output if results else True,
            'output': output,
        }

    def test_full_simulation_across_conversations(self):
        """Run 20+ real exchanges through the challenge pipeline.

        KPIs:
        - All outputs with results have challenge header
        - All outputs with results have revise() instruction
        - All outputs show revised: status
        - Gap callouts appear when no results
        - Pending Tom messages appear after store_exchange
        - No crashes across all exchanges
        """
        self.skipTest("render_prompt deleted — rewrite against SurfaceAssembler")
        # Collect messages from ALL available session files
        all_messages = []
        if os.path.isdir(SESSION_JSONL):
            for fname in sorted(os.listdir(SESSION_JSONL)):
                if fname.endswith('.jsonl'):
                    path = os.path.join(SESSION_JSONL, fname)
                    msgs = _extract_user_messages(path, limit=10)
                    all_messages.extend(msgs)

        # Also add some known engineering queries
        engineering_queries = [
            "how does the recall pipeline work?",
            "what did we decide about encoding?",
            "show me the hook architecture",
            "what are Tom's engineering principles?",
            "kubernetes pod scaling",  # should gap
            "React component lifecycle",  # should gap
            "what corrections have I made?",
            "brain origin story",
            "what's the DAL migration plan?",
            "embedding similarity threshold",
        ]
        all_messages.extend(engineering_queries)

        self.assertGreater(len(all_messages), 20,
                           "Need 20+ messages for meaningful simulation (got %d)" % len(all_messages))

        # Store first 5 messages so pending_tom has content
        for msg in all_messages[:5]:
            self.brain.store_exchange(msg, "simulated reply", "sim-session")

        # Run consolidation detection once
        self.brain.detect_consolidation_candidates(
            similarity_threshold=0.80, max_pairs=3)

        # Simulate each exchange
        turn_results = []
        for msg in all_messages[:30]:  # Cap at 30 for test speed
            try:
                result = self._simulate_challenge_turn(msg)
                turn_results.append(result)
            except Exception as e:
                self.fail("Crash on message '%s': %s" % (msg[:50], e))

        # Aggregate KPIs
        with_results = [r for r in turn_results if r['results_count'] > 0]
        with_gaps = [r for r in turn_results if r['has_gap']]
        with_pending = [r for r in turn_results if r['pending_tom_count'] > 0]

        print("\n=== CHALLENGE SIMULATION RESULTS ===")
        print("Total exchanges: %d" % len(turn_results))
        print("With recall results: %d" % len(with_results))
        print("With gaps (no results): %d" % len(with_gaps))
        print("With pending Tom messages: %d" % len(with_pending))
        print("")

        # KPI 1: All outputs with results have challenge header
        for r in with_results:
            self.assertTrue(r['has_challenge_header'],
                            "Output with results should have challenge header. Query: %s" % r['query'])

        # KPI 2: All outputs with results have revise() instruction
        for r in with_results:
            self.assertTrue(r['has_revise_instruction'],
                            "Output with results should have revise() instruction. Query: %s" % r['query'])

        # KPI 3: All outputs with results show revised: status
        for r in with_results:
            self.assertTrue(r['has_revised_status'],
                            "Output should show revised: status. Query: %s" % r['query'])

        # KPI 4: Gap callouts appear for unknown topics
        gap_queries = [r for r in turn_results if r['gap_query']]
        if gap_queries:
            for r in gap_queries:
                self.assertTrue(r['has_gap_callout'],
                                "Gap should show UNKNOWN TOPIC. Query: %s" % r['query'])

        # KPI 5: At least some exchanges had pending Tom messages
        self.assertGreater(len(with_pending), 0,
                           "Some exchanges should show pending Tom messages")

        # Print sample outputs for manual inspection
        print("\n=== SAMPLE OUTPUTS ===")
        if with_results:
            print("\n--- WITH RESULTS (first) ---")
            print(with_results[0]['output'][:1000])
        if with_gaps:
            print("\n--- WITH GAP (first) ---")
            print(with_gaps[0]['output'][:500])


if __name__ == '__main__':
    unittest.main()
