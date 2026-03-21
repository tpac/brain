"""
brain — Recall Precision Module Tests

Tests for the RecallPrecision class (servers/brain_precision.py) which owns
the entire recall precision lifecycle.

== TEST DESIGN PHILOSOPHY ==

These tests verify the PLUMBING, not the INTELLIGENCE:
  - Data flows correctly through log → evaluate → followup → feedback → summary
  - Schema columns are populated with correct types and values
  - Session isolation works
  - Config-as-state handoff between hooks works
  - Edge cases (empty responses, missing data) don't crash

The actual semantic evaluation (LLM-based) is stubbed with 'pending_llm'.
Tests document what the LLM evaluator iteration should add via FUTURE comments.

== THREE-SIGNAL MODEL ==

Signal 1 — Claude's response: evaluate_response() stores the snippet (stub)
Signal 2 — User's next message: evaluate_followup() stores the message (stub)
Signal 3 — Explicit feedback: receive_feedback() sets precision_score (active)

Only Signal 3 produces precision scores right now. Signals 1 and 2 collect
data for the future LLM evaluator.

== TEST INTEGRITY RULE ==

If any test fails: STOP. Do not change the test OR the code. Report to Tom
what the test expected vs what the code returned. Ask whether the test
expectation is wrong or the code has a bug. Wait for answer before proceeding.
"""

import json
import os
import sys
import unittest

# Ensure parent is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.brain_test_base import BrainTestBase
from servers.brain_precision import RecallPrecision


class TestRecallPrecision(BrainTestBase):
    """Tests for RecallPrecision — the recall precision lifecycle module."""

    def setUp(self):
        super().setUp()

        # Seed realistic brain nodes that mirror the actual brain's content.
        # These are used across tests to simulate real recall scenarios.
        self.auth_rule = self.brain.remember(
            type="rule",
            title="Authentication must use Clerk with magic links",
            content="All auth flows use Clerk. Magic links only. No passwords.",
            locked=True,
        )
        self.adapter_decision = self.brain.remember(
            type="decision",
            title="Supply Adapter pattern for ad delivery",
            content="Ad delivery uses adapter pattern to abstract vendor differences.",
            locked=True,
        )
        self.test_lesson = self.brain.remember(
            type="lesson",
            title="Never weaken test assertions",
            content="When a test fails, stop and ask. Don't change assertEqual to assertGreater.",
            locked=True,
        )
        self.hebbian_concept = self.brain.remember(
            type="concept",
            title="Hebbian learning strengthens co-accessed edges",
            content="Nodes recalled together get stronger connections over time.",
        )
        self.vocab_node = self.brain.remember(
            type="vocabulary",
            title="brain-to-host → communication dimensions",
            content="How brain surfaces information to Claude: consciousness signals, recall injection, encoding prompts",
        )

        # Create precision instance against the same logs DB
        self.precision = RecallPrecision(self.brain.logs_conn, self.brain.conn)

        # Convenience: node IDs for building test data
        self.all_ids = [
            self.auth_rule["id"],
            self.adapter_decision["id"],
            self.test_lesson["id"],
        ]
        self.all_titles = {
            self.auth_rule["id"]: "Authentication must use Clerk with magic links",
            self.adapter_decision["id"]: "Supply Adapter pattern for ad delivery",
            self.test_lesson["id"]: "Never weaken test assertions",
        }
        self.all_snippets = {
            self.auth_rule["id"]: "All auth flows use Clerk. Magic links only.",
            self.adapter_decision["id"]: "Ad delivery uses adapter pattern.",
            self.test_lesson["id"]: "When a test fails, stop and ask.",
        }

    # ── Core Logging Tests ──

    def test_log_recall_basic(self):
        """Log a recall and verify all fields are stored correctly.

        VERIFIES: INSERT into recall_log populates all precision columns.
        SIGNALS: Schema mismatch if this fails — columns may not exist.
        FUTURE: No changes needed — logging is not affected by LLM evaluator.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test1",
            query="how does authentication work",
            returned_ids=self.all_ids,
            recalled_titles=self.all_titles,
            recalled_snippets=self.all_snippets,
            embeddings_used=True,
        )
        self.assertIsNotNone(log_id)
        self.assertGreater(log_id, 0)

        # Verify the row
        row = self.brain.logs_conn.execute(
            "SELECT session_id, query, returned_ids, returned_count, "
            "embeddings_used, recalled_titles, recalled_snippets, created_at "
            "FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "ses_test1")
        self.assertIn("authentication", row[1])
        self.assertEqual(json.loads(row[2]), self.all_ids)
        self.assertEqual(row[3], 3)  # returned_count
        self.assertEqual(row[4], 1)  # embeddings_used = True
        self.assertIsInstance(json.loads(row[5]), dict)  # recalled_titles is valid JSON
        self.assertIsInstance(json.loads(row[6]), dict)  # recalled_snippets is valid JSON
        self.assertIsNotNone(row[7])  # created_at

    def test_log_recall_session_prefix(self):
        """Session IDs are always prefixed with 'ses_'.

        VERIFIES: Prefix enforcement — prevents mixed session_id formats in queries.
        SIGNALS: If this fails, queries filtering by session_id will miss rows.
        FUTURE: No changes needed.
        """
        # Without prefix — should be added
        id1 = self.precision.log_recall(
            session_id="abc123",
            query="test",
            returned_ids=["n1"],
        )
        row1 = self.brain.logs_conn.execute(
            "SELECT session_id FROM recall_log WHERE id = ?", (id1,)
        ).fetchone()
        self.assertEqual(row1[0], "ses_abc123")

        # With prefix — should NOT be double-prefixed
        id2 = self.precision.log_recall(
            session_id="ses_abc123",
            query="test",
            returned_ids=["n2"],
        )
        row2 = self.brain.logs_conn.execute(
            "SELECT session_id FROM recall_log WHERE id = ?", (id2,)
        ).fetchone()
        self.assertEqual(row2[0], "ses_abc123")

    def test_log_recall_without_embeddings(self):
        """Log with embeddings_used=False stores the flag correctly.

        VERIFIES: Degraded recall (keyword-only fallback) is tracked.
        SIGNALS: If this fails, precision summary can't report embeddings usage %.
        FUTURE: No changes needed.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test query",
            returned_ids=["n1"],
            embeddings_used=False,
        )
        row = self.brain.logs_conn.execute(
            "SELECT embeddings_used FROM recall_log WHERE id = ?", (log_id,)
        ).fetchone()
        self.assertEqual(row[0], 0)

    # ── Response Storage Tests (stubs) ──

    def test_evaluate_response_stores_snippet(self):
        """evaluate_response stores Claude's response snippet and marks as pending_llm.

        VERIFIES: Response data is captured for future LLM evaluation.
        SIGNALS: If evaluated_at is not set, the two-turn model can't proceed.
        FUTURE: LLM evaluator should compute used_ids and precision_score here.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="how does auth work",
            returned_ids=[self.auth_rule["id"], self.adapter_decision["id"]],
            recalled_titles=self.all_titles,
            recalled_snippets=self.all_snippets,
            embeddings_used=True,
        )

        result = self.precision.evaluate_response(
            log_id,
            "For authentication, we should use Clerk with magic links as established. "
            "The ad delivery system uses the Supply Adapter pattern to handle vendors.",
        )

        self.assertEqual(result["status"], "stored")
        self.assertGreater(result["assistant_response_len"], 0)

        row = self.brain.logs_conn.execute(
            "SELECT assistant_response_snippet, match_method, evaluated_at, "
            "used_ids, precision_score FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        self.assertIn("Clerk", row[0])  # snippet stored
        self.assertEqual(row[1], "pending_llm")  # stub marker
        self.assertIsNotNone(row[2])  # evaluated_at set
        self.assertIsNone(row[3])  # used_ids still NULL (no LLM yet)
        self.assertIsNone(row[4])  # precision_score still NULL

    def test_evaluate_response_empty_response(self):
        """Empty response doesn't crash — stores gracefully.

        VERIFIES: The error path in hooks (empty last_assistant_message) is handled.
        SIGNALS: If this crashes, the post-response hook will silently fail.
        FUTURE: LLM evaluator should handle empty responses as "no evaluation possible".
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=["n1"],
        )
        result = self.precision.evaluate_response(log_id, "")
        self.assertEqual(result["status"], "stored")
        self.assertEqual(result["assistant_response_len"], 0)

        row = self.brain.logs_conn.execute(
            "SELECT evaluated_at, match_method FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        self.assertIsNotNone(row[0])  # evaluated_at still set
        self.assertEqual(row[1], "pending_llm")

    def test_evaluate_response_long_response_capped(self):
        """Long responses are capped at 500 chars in the snippet.

        VERIFIES: We don't bloat recall_log with huge response text.
        SIGNALS: If snippet > 500 chars, DB rows will grow unbounded.
        FUTURE: LLM evaluator may need the full response — pass it separately.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=["n1"],
        )
        long_response = "x" * 10000
        self.precision.evaluate_response(log_id, long_response)

        row = self.brain.logs_conn.execute(
            "SELECT assistant_response_snippet FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        self.assertLessEqual(len(row[0]), 500)

    # ── Followup Storage Tests (stubs) ──

    def test_evaluate_followup_stores_message(self):
        """evaluate_followup stores user's next message for future LLM analysis.

        VERIFIES: Two-turn model data collection — user message is captured.
        SIGNALS: If evaluation_metadata is empty, followup signal is lost.
        FUTURE: LLM evaluator should analyze whether user builds on or redirects
                from recalled context, replacing 'pending_llm' with real signal.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="how does auth work",
            returned_ids=[self.auth_rule["id"]],
            recalled_titles=self.all_titles,
        )
        self.precision.evaluate_response(log_id, "Use Clerk with magic links.")

        result = self.precision.evaluate_followup(
            log_id, "Perfect, and how does the magic link flow work exactly?"
        )
        self.assertEqual(result["status"], "stored")

        row = self.brain.logs_conn.execute(
            "SELECT followup_signal, evaluation_metadata FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        self.assertEqual(row[0], "pending_llm")
        meta = json.loads(row[1])
        self.assertIn("followup_message", meta)
        self.assertIn("magic link", meta["followup_message"])

    def test_evaluate_followup_negative_message(self):
        """Negative followup message is stored for future LLM analysis.

        VERIFIES: Redirections ("no I meant...") are captured, not lost.
        SIGNALS: If this data isn't stored, the strongest unbiased signal is gone.
        FUTURE: LLM evaluator should detect this as 'negative' signal and
                potentially lower precision_score.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=[self.auth_rule["id"]],
        )
        self.precision.evaluate_response(log_id, "Some response about auth.")

        result = self.precision.evaluate_followup(
            log_id, "No, that's not what I meant. Just fix the typo."
        )
        self.assertEqual(result["status"], "stored")

        row = self.brain.logs_conn.execute(
            "SELECT evaluation_metadata FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        meta = json.loads(row[0])
        self.assertIn("not what I meant", meta["followup_message"])

    def test_evaluate_followup_without_prior_evaluation(self):
        """Followup without prior evaluate_response is handled gracefully.

        VERIFIES: Out-of-order calls don't crash (e.g., if Stop event was missed).
        SIGNALS: If this crashes, a missed Stop event breaks the next turn too.
        FUTURE: LLM evaluator should still store the followup even without response data.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=["n1"],
        )
        # Skip evaluate_response, go straight to followup
        result = self.precision.evaluate_followup(log_id, "Some followup message")
        self.assertEqual(result["status"], "stored")

    # ── Explicit Feedback Tests ──

    def test_receive_feedback_useful(self):
        """'useful' feedback sets precision_score to 1.0.

        VERIFIES: Explicit feedback is the strongest signal and produces real scores.
        SIGNALS: If precision_score isn't 1.0, the feedback override is broken.
        FUTURE: Compare feedback-driven scores vs LLM-computed scores to calibrate.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=[self.auth_rule["id"]],
            recalled_titles=self.all_titles,
        )
        self.precision.evaluate_response(log_id, "Some response")
        self.precision.receive_feedback(log_id, "useful", source="operator")

        row = self.brain.logs_conn.execute(
            "SELECT precision_score, explicit_feedback FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        self.assertEqual(row[0], 1.0)
        fb = json.loads(row[1])
        self.assertEqual(fb["signal"], "useful")
        self.assertEqual(fb["source"], "operator")

    def test_receive_feedback_not_useful(self):
        """'not_useful' feedback sets precision_score to 0.0.

        VERIFIES: Negative feedback correctly zeroes the score.
        SIGNALS: If score isn't 0.0, negative feedback isn't overriding.
        FUTURE: Use accumulated not_useful feedback to identify persistently bad recalls.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=[self.auth_rule["id"]],
        )
        self.precision.receive_feedback(log_id, "not_useful")

        row = self.brain.logs_conn.execute(
            "SELECT precision_score FROM recall_log WHERE id = ?", (log_id,)
        ).fetchone()
        self.assertEqual(row[0], 0.0)

    def test_receive_feedback_partial(self):
        """'partially_useful' feedback sets precision_score to 0.5.

        VERIFIES: Partial feedback maps to the middle of the score range.
        SIGNALS: If score isn't 0.5, the feedback-to-score mapping is wrong.
        FUTURE: Partial feedback is the most interesting for calibration.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=self.all_ids,
        )
        self.precision.receive_feedback(log_id, "partially_useful")

        row = self.brain.logs_conn.execute(
            "SELECT precision_score FROM recall_log WHERE id = ?", (log_id,)
        ).fetchone()
        self.assertEqual(row[0], 0.5)

    # ── Pending Evaluation & Summary Tests ──

    def test_get_pending_evaluation(self):
        """Pending evaluation found before evaluate_response, gone after.

        VERIFIES: The hook can find which recall needs evaluation.
        SIGNALS: If pending returns None when there IS a pending recall, precision loop breaks.
        FUTURE: No changes needed — this is pure plumbing.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="test",
            returned_ids=["n1"],
        )
        # Before evaluation — should be pending
        pending = self.precision.get_pending_evaluation("ses_test")
        self.assertEqual(pending, log_id)

        # After evaluation — should be resolved
        self.precision.evaluate_response(log_id, "some response")
        pending = self.precision.get_pending_evaluation("ses_test")
        self.assertIsNone(pending)

    def test_get_precision_summary_with_feedback(self):
        """Summary aggregates precision scores from explicit feedback.

        VERIFIES: get_precision_summary correctly computes averages and counts.
        SIGNALS: If avg_precision is wrong, consciousness will report bad data.
        FUTURE: Summary should also include LLM-computed scores when available.
        """
        # Log 5 recalls with varying feedback
        feedbacks = ["useful", "not_useful", "useful", "partially_useful", "useful"]
        for i, fb in enumerate(feedbacks):
            log_id = self.precision.log_recall(
                session_id="ses_test",
                query=f"query {i}",
                returned_ids=[f"node_{i}"],
            )
            self.precision.receive_feedback(log_id, fb)

        summary = self.precision.get_precision_summary(hours=1, session_id="ses_test")
        self.assertTrue(summary["has_data"])
        self.assertEqual(summary["total_recalls"], 5)
        self.assertEqual(summary["evaluated_recalls"], 5)
        self.assertEqual(summary["feedback_count"], 5)
        # useful=1.0, not_useful=0.0, useful=1.0, partial=0.5, useful=1.0
        # avg = (1.0 + 0.0 + 1.0 + 0.5 + 1.0) / 5 = 0.7
        self.assertAlmostEqual(summary["avg_precision"], 0.7, places=2)

    def test_get_precision_summary_empty(self):
        """Empty summary when no recalls exist.

        VERIFIES: Summary handles zero-data gracefully.
        SIGNALS: If has_data is True with no data, consciousness signals are wrong.
        FUTURE: No changes needed.
        """
        summary = self.precision.get_precision_summary(hours=1)
        self.assertFalse(summary["has_data"])
        self.assertEqual(summary["total_recalls"], 0)

    def test_request_feedback_returns_message(self):
        """Feedback request returns a formatted string with recalled titles.

        VERIFIES: The feedback mechanism can produce actionable prompts.
        SIGNALS: If this returns None when it should return a message, operator never gets asked.
        FUTURE: Tune the uncertainty threshold once LLM evaluator provides scores.
        """
        log_id = self.precision.log_recall(
            session_id="ses_test",
            query="how does auth work",
            returned_ids=[self.auth_rule["id"]],
            recalled_titles={self.auth_rule["id"]: "Authentication must use Clerk with magic links"},
        )
        self.precision.evaluate_response(log_id, "Some response about auth.")
        # No precision score set (no feedback, no LLM) — should trigger request
        fb = self.precision.request_feedback(log_id)
        self.assertIsNotNone(fb)
        self.assertIn("BRAIN FEEDBACK", fb)
        self.assertIn("Clerk", fb)

    # ── Lifecycle & Infrastructure Tests ──

    def test_full_lifecycle(self):
        """Complete precision lifecycle: log → evaluate → followup → feedback → summary.

        VERIFIES: The entire chain works end-to-end without errors.
        SIGNALS: Any failure here means the precision pipeline is broken.
        FUTURE: With LLM evaluator, verify precision_score changes at each step.
        """
        # 1. Log recall
        log_id = self.precision.log_recall(
            session_id="ses_lifecycle",
            query="authentication and adapters",
            returned_ids=[self.auth_rule["id"], self.adapter_decision["id"]],
            recalled_titles=self.all_titles,
            recalled_snippets=self.all_snippets,
            embeddings_used=True,
        )

        # 2. Evaluate response (stub)
        result = self.precision.evaluate_response(
            log_id,
            "We use Clerk for auth and the adapter pattern for ads.",
        )
        self.assertEqual(result["status"], "stored")

        # 3. Evaluate followup (stub)
        result = self.precision.evaluate_followup(
            log_id,
            "Perfect, now let's look at the webhook integration.",
        )
        self.assertEqual(result["status"], "stored")

        # 4. Receive explicit feedback
        self.precision.receive_feedback(log_id, "useful")

        # 5. Check summary
        summary = self.precision.get_precision_summary(hours=1, session_id="ses_lifecycle")
        self.assertTrue(summary["has_data"])
        self.assertEqual(summary["evaluated_recalls"], 1)
        self.assertEqual(summary["avg_precision"], 1.0)
        self.assertEqual(summary["feedback_count"], 1)

    def test_schema_migration_idempotent(self):
        """Calling _ensure_columns() twice doesn't error.

        VERIFIES: Defensive migration is safe to run multiple times.
        SIGNALS: If this errors, daemon restart could crash.
        FUTURE: No changes needed.
        """
        # First call happened in setUp via RecallPrecision.__init__
        # Second call should be safe
        self.precision._ensure_columns()
        # Third for good measure
        precision2 = RecallPrecision(self.brain.logs_conn, self.brain.conn)
        self.assertIsNotNone(precision2)

    def test_no_double_logging(self):
        """Recall with _skip_log=True doesn't create a recall_log entry.

        VERIFIES: The decoupling works — _log_recall is skipped when hooks handle logging.
        SIGNALS: If count > 0 after _skip_log=True, there's a double-logging bug.
        FUTURE: No changes needed.
        """
        # Count existing recall_log rows
        before = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM recall_log"
        ).fetchone()[0]

        # Call recall with _skip_log=True (simulating what hooks do)
        self.brain.recall("test query", _skip_log=True)

        after = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM recall_log"
        ).fetchone()[0]

        # Should be the same — no new rows from _log_recall
        self.assertEqual(before, after)

    # ── End-to-End Tests ──

    def test_e2e_hook_recall_logs_through_precision(self):
        """Simulate what hook_recall() does: recall → extract → precision.log_recall.

        VERIFIES: The hook → precision → DB path works end-to-end.
        SIGNALS: If recall_log row is missing or has wrong data, the hook wiring is broken.
        FUTURE: Once LLM evaluator exists, extend to verify evaluation after logging.
        """
        # Simulate recall (what hook_recall does)
        result = self.brain.recall("authentication clerk", _skip_log=True)
        results = result.get("results", [])

        # Extract titles/snippets (what hook_recall does)
        recalled_titles = {r.get("id"): r.get("title", "")[:100] for r in results}
        recalled_snippets = {r.get("id"): (r.get("content") or "")[:150] for r in results}

        # Log through precision (what hook_recall does)
        log_id = self.precision.log_recall(
            session_id="ses_e2e",
            query="authentication clerk",
            returned_ids=[r.get("id") for r in results],
            recalled_titles=recalled_titles,
            recalled_snippets=recalled_snippets,
            embeddings_used=False,  # keyword-only in test
        )

        # Verify exactly 1 row
        count = self.brain.logs_conn.execute(
            "SELECT COUNT(*) FROM recall_log WHERE session_id = 'ses_e2e'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

        # Verify titles match
        row = self.brain.logs_conn.execute(
            "SELECT recalled_titles, embeddings_used FROM recall_log WHERE id = ?",
            (log_id,),
        ).fetchone()
        stored_titles = json.loads(row[0])
        for nid, title in recalled_titles.items():
            self.assertEqual(stored_titles.get(nid), title)
        self.assertEqual(row[1], 0)  # embeddings_used = False

    def test_e2e_full_two_turn_cycle(self):
        """Simulate a complete two-turn precision cycle via config-as-state.

        VERIFIES: The full hook lifecycle with config handoff between hook_recall
        and hook_post_response_track works correctly.
        SIGNALS: If config keys are stale or overwritten, turns will cross-contaminate.
        FUTURE: With LLM evaluator, verify precision_score and used_ids are computed.
        """
        # ── Turn 1 ──
        # hook_recall fires → log recall
        log_id_1 = self.precision.log_recall(
            session_id="ses_2turn",
            query="how does auth work",
            returned_ids=[self.auth_rule["id"]],
            recalled_titles=self.all_titles,
            embeddings_used=True,
        )
        self.brain.set_config("last_recall_log_id", str(log_id_1))

        # hook_post_response_track fires → evaluate response
        recall_log_id = self.brain.get_config("last_recall_log_id", "")
        self.assertEqual(recall_log_id, str(log_id_1))
        self.precision.evaluate_response(int(recall_log_id), "Use Clerk for auth.")
        self.brain.set_config("last_evaluated_recall_id", recall_log_id)
        self.brain.set_config("last_recall_log_id", "")  # clear

        # ── Turn 2 ──
        # hook_recall fires → evaluate followup from turn 1
        prev_eval_id = self.brain.get_config("last_evaluated_recall_id", "")
        self.assertEqual(prev_eval_id, str(log_id_1))
        self.precision.evaluate_followup(int(prev_eval_id), "Perfect, show me the webhook flow")
        self.brain.set_config("last_evaluated_recall_id", "")

        # Log new recall for turn 2
        log_id_2 = self.precision.log_recall(
            session_id="ses_2turn",
            query="webhook flow",
            returned_ids=[self.adapter_decision["id"]],
            recalled_titles=self.all_titles,
            embeddings_used=True,
        )
        self.brain.set_config("last_recall_log_id", str(log_id_2))

        # Verify turn 1 row has all data
        row1 = self.brain.logs_conn.execute(
            "SELECT assistant_response_snippet, followup_signal, evaluation_metadata "
            "FROM recall_log WHERE id = ?",
            (log_id_1,),
        ).fetchone()
        self.assertIn("Clerk", row1[0])  # response stored
        self.assertEqual(row1[1], "pending_llm")  # followup stored
        meta = json.loads(row1[2])
        self.assertIn("webhook", meta["followup_message"])

        # Verify turn 2 is a fresh row
        row2 = self.brain.logs_conn.execute(
            "SELECT assistant_response_snippet, followup_signal FROM recall_log WHERE id = ?",
            (log_id_2,),
        ).fetchone()
        self.assertIsNone(row2[0])  # not evaluated yet
        self.assertIsNone(row2[1])  # no followup yet

        # Verify config state
        self.assertEqual(self.brain.get_config("last_recall_log_id", ""), str(log_id_2))
        self.assertEqual(self.brain.get_config("last_evaluated_recall_id", ""), "")

    def test_e2e_feedback_overrides_after_full_cycle(self):
        """Feedback overrides precision after full cycle, summary reflects it.

        VERIFIES: Explicit feedback is the strongest signal and correctly
        overrides the pending_llm state. Summary aggregation works with mixed data.
        SIGNALS: If summary avg is wrong, the feedback-to-summary pipeline is broken.
        FUTURE: Compare feedback-driven scores vs LLM-computed scores to calibrate.
        """
        # Cycle 1: log → evaluate → followup → feedback (not_useful)
        id1 = self.precision.log_recall(
            session_id="ses_fb",
            query="q1",
            returned_ids=["n1"],
            recalled_titles={"n1": "Node 1"},
        )
        self.precision.evaluate_response(id1, "Response about node 1")
        self.precision.evaluate_followup(id1, "That's not what I needed")
        self.precision.receive_feedback(id1, "not_useful")

        # Cycle 2: log → feedback (useful)
        id2 = self.precision.log_recall(
            session_id="ses_fb",
            query="q2",
            returned_ids=["n2"],
            recalled_titles={"n2": "Node 2"},
        )
        self.precision.receive_feedback(id2, "useful")

        # Summary
        summary = self.precision.get_precision_summary(hours=1, session_id="ses_fb")
        self.assertEqual(summary["total_recalls"], 2)
        self.assertEqual(summary["evaluated_recalls"], 2)
        self.assertEqual(summary["feedback_count"], 2)
        # avg = (0.0 + 1.0) / 2 = 0.5
        self.assertAlmostEqual(summary["avg_precision"], 0.5, places=2)

    def test_e2e_empty_response_error_path(self):
        """Empty response stores evaluation but hooks should log diagnostics.

        VERIFIES: The error path doesn't crash and evaluation still proceeds.
        SIGNALS: If this crashes, a single empty Stop event breaks the loop.
        FUTURE: Track frequency of empty responses to detect hook communication issues.
        """
        log_id = self.precision.log_recall(
            session_id="ses_empty",
            query="test",
            returned_ids=["n1"],
        )

        # evaluate_response with empty — should not crash
        result = self.precision.evaluate_response(log_id, "")
        self.assertEqual(result["status"], "stored")

        # Simulate the hook's error logging (what hook_post_response_track does)
        self.brain._logs_dal.write_debug(
            "precision",
            "Empty/short assistant response in Stop event",
            session_id="ses_empty",
            metadata=json.dumps({"response_len": 0, "recall_log_id": str(log_id)}),
        )

        # Verify debug_log has the diagnostic entry
        diag = self.brain.logs_conn.execute(
            "SELECT source, metadata FROM debug_log WHERE source = 'precision'"
        ).fetchone()
        self.assertIsNotNone(diag)
        self.assertEqual(diag[0], "precision")

    def test_e2e_multiple_sessions_isolation(self):
        """Sessions are isolated — each sees only its own data.

        VERIFIES: session_id filtering works correctly in queries.
        SIGNALS: If sessions bleed into each other, precision data is corrupted.
        FUTURE: Cross-session precision trends (is recall getting better over time?).
        """
        # Session A
        id_a = self.precision.log_recall(
            session_id="ses_session_a",
            query="query A",
            returned_ids=["na1"],
        )
        self.precision.receive_feedback(id_a, "useful")

        # Session B
        id_b = self.precision.log_recall(
            session_id="ses_session_b",
            query="query B",
            returned_ids=["nb1"],
        )
        self.precision.receive_feedback(id_b, "not_useful")

        # Pending evaluations are session-scoped
        # (both have been evaluated via feedback, so pending = None)
        # But let's add an unevaluated recall to session A
        id_a2 = self.precision.log_recall(
            session_id="ses_session_a",
            query="query A2",
            returned_ids=["na2"],
        )
        pending_a = self.precision.get_pending_evaluation("ses_session_a")
        pending_b = self.precision.get_pending_evaluation("ses_session_b")
        self.assertEqual(pending_a, id_a2)
        self.assertIsNone(pending_b)

        # Summaries are session-scoped
        summary_a = self.precision.get_precision_summary(hours=1, session_id="ses_session_a")
        summary_b = self.precision.get_precision_summary(hours=1, session_id="ses_session_b")
        self.assertEqual(summary_a["total_recalls"], 2)  # id_a + id_a2
        self.assertEqual(summary_a["evaluated_recalls"], 1)  # only id_a has score
        self.assertEqual(summary_b["total_recalls"], 1)
        self.assertEqual(summary_b["evaluated_recalls"], 1)
        self.assertEqual(summary_b["avg_precision"], 0.0)  # not_useful


if __name__ == "__main__":
    unittest.main()
