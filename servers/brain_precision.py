"""
brain — Recall Precision Module

Owns the entire recall precision lifecycle. No other module should read/write
the recall_log table directly — all precision data flows through this class.

== WHY THIS EXISTS ==

The brain recalls nodes before Claude responds, but previously never learned
whether those nodes were useful. The recall_log table had used_ids, used_count,
precision_score columns that were always NULL/0 because mark_recall_used() was
never implemented. This module closes that feedback loop.

== THREE-SIGNAL EVALUATION MODEL ==

Signal 1 — Claude's response (WEAK, BIASED):
    Claude absorbs injected context and naturally references it, even when
    irrelevant. This creates a reinforcement bias: popular nodes get recalled
    → mentioned → scored as "useful" → recalled more. Runaway positive feedback.
    We store the response for future LLM-based analysis but do NOT auto-score.

Signal 2 — User's next message (STRONG, UNBIASED):
    The user didn't see the raw recall injection. Their follow-up reflects
    genuine usefulness: building on recalled concepts = positive, redirecting
    ("no I meant...") = negative. This is the two-turn evaluation model.
    Stored for future LLM analysis.

Signal 3 — Explicit operator feedback (STRONGEST):
    Direct confirmation when the system is uncertain. The operator can say
    "useful", "not_useful", or "partially_useful". This overrides any
    computed scores and provides ground-truth calibration data.

== LAYERED EVALUATION (v3) ==

evaluate_response() stores response data and pre-computes embedding signals.
evaluate_followup() runs the full layered evaluation:
  L0: Expanded regex (30+ patterns) — intent markers
  L1: Arctic embeddings — topic similarity
  L1b: BART-Large-MNLI — stance detection (if available)
  L2: Cross-signal patterns — combos, parroting, polite dismiss

Validated at 92% accuracy (11/12 clear cases) via 10 experiments.
See docs/PRECISION-EVAL-RESULTS.md for full evaluation history.

The one remaining miss (semantic contradiction — user advocates anti-pattern)
requires LLM. Deferred to future Haiku integration.

== WHAT WE STORE AND WHY ==

We store recalled_titles and recalled_snippets redundantly (the node data
exists in brain.db) because:
  - Nodes may change or be deleted between recall and analysis
  - We need the exact content that was surfaced to Claude for accurate evaluation
  - This makes the recall_log self-contained for auditing and debugging
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── Column definitions for defensive migration ──
# Each tuple: (column_name, SQL type + default)
# These match schema.py's LOG_TABLES['recall_log'] and migration 004.
_PRECISION_COLUMNS = [
    ("embeddings_used", "INTEGER DEFAULT 0"),
    ("recalled_titles", "TEXT"),
    ("recalled_snippets", "TEXT"),
    ("assistant_response_snippet", "TEXT"),
    ("match_method", "TEXT"),
    ("evaluation_metadata", "TEXT"),
    ("followup_signal", "TEXT"),
    ("explicit_feedback", "TEXT"),
    ("evaluated_at", "TEXT"),
]

# Feedback signal → precision score mapping.
# Explicit feedback is the strongest signal and directly sets the score.
_FEEDBACK_SCORES = {
    "useful": 1.0,
    "not_useful": 0.0,
    "partially_useful": 0.5,
}

# Maximum snippet lengths to prevent bloat in recall_log rows.
_MAX_RESPONSE_SNIPPET = 500
_MAX_FOLLOWUP_SNIPPET = 500
_MAX_TITLE_LEN = 100
_MAX_SNIPPET_LEN = 150

# Cooldown: don't request feedback more than once every N recall events.
_FEEDBACK_COOLDOWN = 5


class RecallPrecision:
    """
    Owns the entire recall precision lifecycle.

    Constructor takes logs_conn (brain_logs.db) and optionally brain_conn
    (brain.db, for future use by LLM evaluator to fetch full node context).

    Usage from hooks:
        precision = RecallPrecision(brain.logs_conn, brain.conn)
        log_id = precision.log_recall(session_id, query, returned_ids, ...)
        precision.evaluate_response(log_id, assistant_response)
        precision.evaluate_followup(log_id, user_next_message)
        precision.receive_feedback(log_id, "useful")
        summary = precision.get_precision_summary(hours=168)
    """

    def __init__(self, logs_conn: sqlite3.Connection, brain_conn: Optional[sqlite3.Connection] = None):
        self.logs_conn = logs_conn
        self.brain_conn = brain_conn
        self._ensure_columns()

    # ── Schema Safety ──

    def _ensure_columns(self) -> None:
        """Defensively add precision columns if they don't exist yet.

        WHY: Migration 004 handles existing databases, and schema.py handles
        fresh databases. But if neither has run yet (e.g., Brain initialized
        before migrations), this ensures the columns exist. Each ALTER is
        idempotent — duplicate ADD COLUMN raises OperationalError which we catch.
        """
        for col_name, col_def in _PRECISION_COLUMNS:
            try:
                self.logs_conn.execute(
                    f"ALTER TABLE recall_log ADD COLUMN {col_name} {col_def}"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists
        try:
            self.logs_conn.commit()
        except Exception:
            pass

    # ── Core: Log Recall ──

    def log_recall(
        self,
        session_id: str,
        query: str,
        returned_ids: List[str],
        recalled_titles: Optional[Dict[str, str]] = None,
        recalled_snippets: Optional[Dict[str, str]] = None,
        embeddings_used: bool = True,
    ) -> int:
        """Record a recall event with full context for future evaluation.

        This is the entry point for the precision lifecycle. Called by
        hook_recall() after brain.recall_with_embeddings() returns results.

        We store titles and snippets redundantly so the recall_log is
        self-contained for auditing — nodes may change or be deleted
        between recall and evaluation.

        Args:
            session_id: Session identifier. Will be prefixed with 'ses_' if not already.
            query: The enriched query that was used for recall.
            returned_ids: List of node IDs that were returned.
            recalled_titles: {node_id: title} for each returned node.
            recalled_snippets: {node_id: content_snippet} for each returned node.
            embeddings_used: True if embedding recall was used, False if keyword-only fallback.

        Returns:
            The recall_log row ID.
        """
        # Enforce ses_ prefix — good practice for session isolation and future queries.
        if not session_id.startswith("ses_"):
            session_id = f"ses_{session_id}"

        now = datetime.now(timezone.utc).isoformat()

        # Truncate titles and snippets to prevent bloat
        titles_json = json.dumps(
            {nid: t[:_MAX_TITLE_LEN] for nid, t in (recalled_titles or {}).items()}
        )
        snippets_json = json.dumps(
            {nid: s[:_MAX_SNIPPET_LEN] for nid, s in (recalled_snippets or {}).items()}
        )

        cursor = self.logs_conn.execute(
            """INSERT INTO recall_log
               (session_id, query, returned_ids, returned_count,
                embeddings_used, recalled_titles, recalled_snippets, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                query[:500],
                json.dumps(returned_ids),
                len(returned_ids),
                1 if embeddings_used else 0,
                titles_json,
                snippets_json,
                now,
            ),
        )
        self.logs_conn.commit()
        return cursor.lastrowid

    # ── Signal 1: Claude's Response ──

    def evaluate_response(self, recall_log_id: int, assistant_response: str) -> Dict[str, Any]:
        """Store Claude's response and pre-compute embedding signals.

        The response is a BIASED signal (Claude absorbs injected context),
        so we do NOT compute precision_score here. We store the snippet and
        pre-compute embedding signals that evaluate_followup() will use.

        Args:
            recall_log_id: The recall_log row ID from log_recall().
            assistant_response: Claude's full response text.

        Returns:
            Dict with recall_log_id, status, and response length.
        """
        now = datetime.now(timezone.utc).isoformat()
        snippet = (assistant_response or "")[:_MAX_RESPONSE_SNIPPET]

        # Pre-compute embedding signals if response is substantial
        eval_meta = {}
        if snippet and len(snippet) >= 20:
            row = self.logs_conn.execute(
                "SELECT recalled_titles, recalled_snippets FROM recall_log WHERE id = ?",
                (recall_log_id,),
            ).fetchone()
            if row and row[0] and row[1]:
                try:
                    titles = json.loads(row[0])
                    snippets = json.loads(row[1])
                    from servers.recall_scorer import compute_embedding_signals
                    emb_signals = compute_embedding_signals(snippets, titles, snippet, "")
                    eval_meta["response_emb_signals"] = emb_signals
                except Exception:
                    pass

        meta_json = json.dumps(eval_meta) if eval_meta else None

        self.logs_conn.execute(
            """UPDATE recall_log
               SET assistant_response_snippet = ?,
                   match_method = 'response_stored',
                   evaluation_metadata = COALESCE(?, evaluation_metadata),
                   evaluated_at = ?
               WHERE id = ?""",
            (snippet, meta_json, now, recall_log_id),
        )
        self.logs_conn.commit()

        return {
            "recall_log_id": recall_log_id,
            "status": "stored",
            "assistant_response_len": len(assistant_response or ""),
        }

    # ── Signal 2: User's Next Message (layered evaluation) ──

    def evaluate_followup(self, recall_log_id: int, user_message: str) -> Dict[str, Any]:
        """Evaluate user's follow-up message using layered scoring.

        This is the UNBIASED signal — the user didn't see the raw recall
        injection. Their follow-up reflects genuine usefulness.

        Runs L0 (regex) + L1 (embeddings, if available) + L1b (BART, if
        available) + L2 (cross-signal patterns). Writes followup_signal,
        match_method, precision_score, and evaluation_metadata.

        Never overrides explicit_feedback — operator feedback is ground truth.

        Args:
            recall_log_id: The recall_log row ID from log_recall().
            user_message: The user's message in the NEXT turn after recall.

        Returns:
            Dict with recall_log_id, status, followup_signal, evidence, confidence.
        """
        now = datetime.now(timezone.utc).isoformat()
        followup_snippet = (user_message or "")[:_MAX_FOLLOWUP_SNIPPET]

        # Fetch existing data
        row = self.logs_conn.execute(
            """SELECT evaluation_metadata, recalled_titles, recalled_snippets,
                      assistant_response_snippet, explicit_feedback
               FROM recall_log WHERE id = ?""",
            (recall_log_id,),
        ).fetchone()

        if not row:
            return {"recall_log_id": recall_log_id, "status": "not_found"}

        existing_meta = {}
        if row[0]:
            try:
                existing_meta = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                pass

        existing_meta["followup_message"] = followup_snippet
        existing_meta["followup_stored_at"] = now

        # Don't override explicit feedback
        if row[4]:
            existing_meta["auto_eval_skipped"] = "explicit_feedback_exists"
            self.logs_conn.execute(
                """UPDATE recall_log
                   SET evaluation_metadata = ?, evaluated_at = ?
                   WHERE id = ?""",
                (json.dumps(existing_meta), now, recall_log_id),
            )
            self.logs_conn.commit()
            return {
                "recall_log_id": recall_log_id,
                "status": "skipped_has_feedback",
                "followup_len": len(user_message or ""),
            }

        # Parse recalled data
        titles = {}
        snippets = {}
        response = row[3] or ""
        if row[1]:
            try:
                titles = json.loads(row[1])
            except (json.JSONDecodeError, TypeError):
                pass
        if row[2]:
            try:
                snippets = json.loads(row[2])
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Run layered evaluation ──
        from servers.recall_scorer import (
            compute_regex_signals, compute_bart_signals,
            compute_embedding_signals, score_recall,
            evidence_to_precision, classify_followup_signal,
            determine_match_method, is_bart_ready,
        )

        regex_signals = compute_regex_signals(followup_snippet)
        bart_signals = compute_bart_signals(followup_snippet)

        # Embedding signals: use pre-computed from evaluate_response if available,
        # otherwise compute fresh
        emb_signals = existing_meta.get("response_emb_signals")
        emb_available = False
        if snippets and titles:
            try:
                emb_signals = compute_embedding_signals(
                    snippets, titles, response, followup_snippet,
                )
                emb_available = True
            except Exception:
                pass

        if not emb_signals:
            from servers.recall_scorer import _empty_emb
            emb_signals = _empty_emb()
            # Carry over resp_words from response if we have it
            if response:
                emb_signals["resp_words"] = len(response.split())

        evidence, confidence, reasons, n_signals = score_recall(
            regex_signals, bart_signals, emb_signals,
        )
        precision = evidence_to_precision(evidence, confidence)
        signal = classify_followup_signal(evidence, confidence)
        method = determine_match_method(is_bart_ready(), emb_available)

        # Store evaluation results
        top_reasons = [(w, r, d) for w, r, d in reasons[:5]]
        existing_meta["evidence"] = round(evidence, 4)
        existing_meta["confidence"] = round(confidence, 4)
        existing_meta["n_signals"] = n_signals
        existing_meta["top_reasons"] = top_reasons
        existing_meta["regex_signals"] = regex_signals
        existing_meta["bart_available"] = is_bart_ready()
        existing_meta["emb_available"] = emb_available

        self.logs_conn.execute(
            """UPDATE recall_log
               SET followup_signal = ?,
                   match_method = ?,
                   precision_score = ?,
                   evaluation_metadata = ?,
                   evaluated_at = ?
               WHERE id = ? AND explicit_feedback IS NULL""",
            (signal, method, precision, json.dumps(existing_meta), now, recall_log_id),
        )
        self.logs_conn.commit()

        return {
            "recall_log_id": recall_log_id,
            "status": "stored",
            "followup_signal": signal,
            "evidence": round(evidence, 4),
            "confidence": round(confidence, 4),
            "followup_len": len(user_message or ""),
        }

    # ── Signal 3: Explicit Operator Feedback ──

    def receive_feedback(
        self, recall_log_id: int, feedback: str, source: str = "operator"
    ) -> None:
        """Process explicit operator feedback — the strongest signal.

        This overrides any computed precision scores. When the operator says
        "not_useful", that's ground truth regardless of what substring
        matching or LLM evaluation might say.

        Feedback values:
          - "useful" → precision_score = 1.0
          - "not_useful" → precision_score = 0.0
          - "partially_useful" → precision_score = 0.5

        Args:
            recall_log_id: The recall_log row ID.
            feedback: One of "useful", "not_useful", "partially_useful".
            source: Who provided the feedback (default "operator").
        """
        now = datetime.now(timezone.utc).isoformat()
        score = _FEEDBACK_SCORES.get(feedback, 0.5)

        feedback_json = json.dumps({
            "signal": feedback,
            "source": source,
            "timestamp": now,
        })

        self.logs_conn.execute(
            """UPDATE recall_log
               SET explicit_feedback = ?,
                   precision_score = ?,
                   evaluated_at = ?
               WHERE id = ?""",
            (feedback_json, score, now, recall_log_id),
        )
        self.logs_conn.commit()

    # ── Feedback Request ──

    def request_feedback(self, recall_log_id: int) -> Optional[str]:
        """Check if we should ask the operator for feedback on this recall.

        Only requests feedback when:
          - The recall exists and has been through evaluate_response
          - No explicit feedback has been given yet
          - No LLM-computed precision score exists yet
          - We're not on cooldown (don't spam the operator)

        Returns:
            A formatted feedback request string, or None if not needed.
        """
        if not recall_log_id or recall_log_id <= 0:
            return None

        row = self.logs_conn.execute(
            """SELECT returned_count, recalled_titles, explicit_feedback,
                      precision_score, evaluated_at
               FROM recall_log WHERE id = ?""",
            (recall_log_id,),
        ).fetchone()

        if not row:
            return None

        returned_count, titles_json, explicit_fb, precision, evaluated_at = row

        # Don't ask if already has explicit feedback
        if explicit_fb:
            return None

        # Don't ask if precision already computed with high confidence
        if precision is not None:
            # Check confidence in metadata — only skip if confident
            try:
                meta_row = self.logs_conn.execute(
                    "SELECT evaluation_metadata FROM recall_log WHERE id = ?",
                    (recall_log_id,),
                ).fetchone()
                if meta_row and meta_row[0]:
                    meta = json.loads(meta_row[0])
                    if meta.get("confidence", 1.0) >= 0.5:
                        return None
            except Exception:
                return None

        # Don't ask if no recall happened
        if not returned_count or returned_count == 0:
            return None

        # Cooldown: check how many recalls since last feedback request
        recent_count = self.logs_conn.execute(
            """SELECT COUNT(*) FROM recall_log
               WHERE id > ? AND explicit_feedback IS NOT NULL""",
            (max(0, recall_log_id - _FEEDBACK_COOLDOWN),),
        ).fetchone()[0]
        # If there was recent feedback, skip
        if recent_count > 0:
            return None

        # Build the feedback request message
        titles = []
        if titles_json:
            try:
                titles_dict = json.loads(titles_json)
                titles = list(titles_dict.values())[:5]
            except (json.JSONDecodeError, TypeError):
                pass

        if not titles:
            return None

        title_list = "\n".join(f"  - {t}" for t in titles)
        return (
            f"BRAIN FEEDBACK REQUEST: {returned_count} node(s) were recalled. "
            f"Were they relevant?\n{title_list}\n"
            f"(Reply: useful / not_useful / partially_useful — or ignore to skip)"
        )

    # ── Query Methods ──

    def get_pending_evaluation(self, session_id: str) -> Optional[int]:
        """Find the most recent recall that hasn't been evaluated yet.

        Used by hook_recall() to check if the previous turn's recall
        needs followup evaluation before starting a new recall cycle.

        Args:
            session_id: The current session ID.

        Returns:
            The recall_log row ID, or None if no pending evaluation.
        """
        # Enforce prefix consistency
        if not session_id.startswith("ses_"):
            session_id = f"ses_{session_id}"

        row = self.logs_conn.execute(
            """SELECT id FROM recall_log
               WHERE session_id = ? AND evaluated_at IS NULL AND returned_count > 0
               ORDER BY created_at DESC LIMIT 1""",
            (session_id,),
        ).fetchone()

        return row[0] if row else None

    def get_precision_summary(
        self, hours: int = 24, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Aggregate precision statistics for consciousness signals.

        This is what consciousness reads to decide whether recall quality
        is healthy, degrading, or needs attention. Consciousness does NOT
        own precision logic — it reads this summary and decides what to surface.

        Args:
            hours: Look-back window in hours (default 24).
            session_id: Optional filter by session. If None, all sessions.

        Returns:
            Dict with total_recalls, evaluated_recalls, avg_precision,
            embeddings_used_pct, has_data, total_returned_nodes,
            followup_signals counts, and feedback_count.
        """
        cutoff = datetime.now(timezone.utc).isoformat()
        # Approximate cutoff by subtracting hours (good enough for summary)
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        base_where = "WHERE created_at > ?"
        params: list = [cutoff]

        if session_id:
            if not session_id.startswith("ses_"):
                session_id = f"ses_{session_id}"
            base_where += " AND session_id = ?"
            params.append(session_id)

        # Total recalls
        total = self.logs_conn.execute(
            f"SELECT COUNT(*) FROM recall_log {base_where}", params
        ).fetchone()[0]

        if total == 0:
            return {
                "total_recalls": 0,
                "evaluated_recalls": 0,
                "avg_precision": 0.0,
                "embeddings_used_pct": 0.0,
                "has_data": False,
                "total_returned_nodes": 0,
                "followup_signals": {"positive": 0, "negative": 0, "neutral": 0, "pending_llm": 0},
                "feedback_count": 0,
            }

        # Evaluated (has precision_score from feedback)
        evaluated = self.logs_conn.execute(
            f"SELECT COUNT(*) FROM recall_log {base_where} AND precision_score IS NOT NULL",
            params,
        ).fetchone()[0]

        # Average precision (only from rows with scores)
        avg_row = self.logs_conn.execute(
            f"SELECT AVG(precision_score) FROM recall_log {base_where} AND precision_score IS NOT NULL",
            params,
        ).fetchone()
        avg_precision = avg_row[0] if avg_row and avg_row[0] is not None else 0.0

        # Embeddings usage
        emb_count = self.logs_conn.execute(
            f"SELECT COUNT(*) FROM recall_log {base_where} AND embeddings_used = 1",
            params,
        ).fetchone()[0]

        # Total returned nodes
        total_returned = self.logs_conn.execute(
            f"SELECT SUM(returned_count) FROM recall_log {base_where}",
            params,
        ).fetchone()[0] or 0

        # Followup signals breakdown
        followup_signals = {"positive": 0, "negative": 0, "neutral": 0, "uncertain": 0, "pending_llm": 0}
        for signal_val in followup_signals:
            count = self.logs_conn.execute(
                f"SELECT COUNT(*) FROM recall_log {base_where} AND followup_signal = ?",
                params + [signal_val],
            ).fetchone()[0]
            followup_signals[signal_val] = count

        # Explicit feedback count
        feedback_count = self.logs_conn.execute(
            f"SELECT COUNT(*) FROM recall_log {base_where} AND explicit_feedback IS NOT NULL",
            params,
        ).fetchone()[0]

        return {
            "total_recalls": total,
            "evaluated_recalls": evaluated,
            "avg_precision": round(avg_precision, 3),
            "embeddings_used_pct": round(emb_count / total, 3) if total > 0 else 0.0,
            "has_data": True,
            "total_returned_nodes": total_returned,
            "followup_signals": followup_signals,
            "feedback_count": feedback_count,
        }
