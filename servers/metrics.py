"""
brain — Validation Metrics & Feature Effectiveness Telemetry

Tracks which brain features actually help recall and improve sessions.
All metrics go to brain_logs.db (NOT brain.db) via the debug_log table.

Usage:
    metrics = BrainMetrics(logs_conn)
    metrics.record_recall_hit(query, node_id, source="embedding")
    metrics.record_recall_miss(query)
    metrics.record_signal_surfaced("fading_knowledge")
    metrics.record_heartbeat_nudge(severity="gentle", encoded_after=True)

    report = metrics.get_effectiveness_report()
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class BrainMetrics:
    """Feature effectiveness tracker — measures what actually helps."""

    def __init__(self, logs_conn):
        """
        Args:
            logs_conn: SQLite connection to brain_logs.db
        """
        self.conn = logs_conn
        self._ensure_tables()

    def _ensure_tables(self):
        """Create metrics tables if they don't exist."""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feature_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                feature TEXT NOT NULL,
                value REAL DEFAULT 1.0,
                metadata TEXT,
                session_id TEXT
            )
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_feature_metrics_type
            ON feature_metrics(metric_type, feature)
        ''')
        self.conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_feature_metrics_time
            ON feature_metrics(timestamp)
        ''')
        self.conn.commit()

    def _now(self) -> str:
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    def _record(self, metric_type: str, feature: str, value: float = 1.0,
                metadata: Optional[Dict] = None, session_id: Optional[str] = None):
        """Record a metric event."""
        try:
            self.conn.execute(
                '''INSERT INTO feature_metrics (timestamp, metric_type, feature, value, metadata, session_id)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (self._now(), metric_type, feature, value,
                 json.dumps(metadata) if metadata else None, session_id)
            )
            self.conn.commit()
        except Exception:
            pass

    # ─── Recording Methods ───

    def record_recall_hit(self, query: str, node_id: str, source: str = "embedding",
                         score: float = 0.0):
        """A recalled node was useful (user acted on it or referenced it)."""
        self._record("recall_hit", source, score, {
            "query": query[:200], "node_id": node_id
        })

    def record_recall_miss(self, query: str, source: str = "embedding"):
        """Recall returned nothing useful for this query."""
        self._record("recall_miss", source, 0.0, {"query": query[:200]})

    def record_recall_attempt(self, query: str, num_results: int, source: str = "embedding",
                             top_score: float = 0.0):
        """Track every recall attempt for hit rate calculation."""
        self._record("recall_attempt", source, top_score, {
            "query": query[:200], "num_results": num_results
        })

    def record_signal_surfaced(self, signal_type: str, count: int = 1):
        """A consciousness signal was shown at boot or during session."""
        self._record("signal_surfaced", signal_type, float(count))

    def record_signal_acted_on(self, signal_type: str, action: str = "acknowledged"):
        """User or Claude acted on a consciousness signal."""
        self._record("signal_acted_on", signal_type, 1.0, {"action": action})

    def record_dream_connection_used(self, edge_id: str, query: str):
        """A dream-created edge appeared in recall results."""
        self._record("dream_used", "dream_edge", 1.0, {
            "edge_id": edge_id, "query": query[:200]
        })

    def record_vocab_resolved(self, term: str, context: Optional[str] = None):
        """Vocabulary system resolved a term during recall enrichment."""
        self._record("vocab_resolved", "vocabulary", 1.0, {
            "term": term, "context": context[:100] if context else None
        })

    def record_heartbeat_nudge(self, severity: str, messages_since: int,
                              encoded_after: bool = False):
        """Heartbeat fired a nudge. Track if encoding followed."""
        self._record("heartbeat_nudge", severity, 1.0 if encoded_after else 0.0, {
            "messages_since_encode": messages_since,
            "encoded_after": encoded_after
        })

    def record_heartbeat_result(self, encoded_after: bool):
        """Update the last heartbeat nudge with whether encoding happened."""
        if encoded_after:
            self._record("heartbeat_effective", "heartbeat", 1.0)
        else:
            self._record("heartbeat_ignored", "heartbeat", 0.0)

    def record_feature_usage(self, feature: str, action: str = "used"):
        """Generic feature usage counter."""
        self._record("feature_usage", feature, 1.0, {"action": action})

    def record_daemon_speedup(self, operation: str, cold_ms: float, warm_ms: float):
        """Track daemon vs cold-start performance."""
        self._record("daemon_speedup", operation, cold_ms / max(warm_ms, 1), {
            "cold_ms": cold_ms, "warm_ms": warm_ms
        })

    # ─── Reporting Methods ───

    def get_effectiveness_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Compute feature effectiveness metrics over the given period.
        Returns a report suitable for consciousness signal display.
        """
        cutoff = "datetime('now', '-%d days')" % days
        report = {}

        # Recall hit rate
        try:
            attempts = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'recall_attempt' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            hits = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'recall_hit' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            misses = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'recall_miss' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            report['recall'] = {
                'attempts': attempts,
                'hits': hits,
                'misses': misses,
                'hit_rate': hits / max(attempts, 1),
            }
        except Exception:
            report['recall'] = {'attempts': 0, 'hits': 0, 'misses': 0, 'hit_rate': 0}

        # Signal engagement
        try:
            surfaced = self.conn.execute(
                "SELECT feature, SUM(value) FROM feature_metrics WHERE metric_type = 'signal_surfaced' AND timestamp > %s GROUP BY feature" % cutoff
            ).fetchall()
            acted = self.conn.execute(
                "SELECT feature, COUNT(*) FROM feature_metrics WHERE metric_type = 'signal_acted_on' AND timestamp > %s GROUP BY feature" % cutoff
            ).fetchall()
            surfaced_dict = {r[0]: r[1] for r in surfaced}
            acted_dict = {r[0]: r[1] for r in acted}
            signal_report = {}
            for signal_type in set(list(surfaced_dict.keys()) + list(acted_dict.keys())):
                s = surfaced_dict.get(signal_type, 0)
                a = acted_dict.get(signal_type, 0)
                signal_report[signal_type] = {
                    'surfaced': int(s),
                    'acted_on': int(a),
                    'engagement_rate': a / max(s, 1),
                }
            report['signals'] = signal_report
        except Exception:
            report['signals'] = {}

        # Dream connection utility
        try:
            dream_used = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'dream_used' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            report['dreams'] = {'connections_used': dream_used}
        except Exception:
            report['dreams'] = {'connections_used': 0}

        # Vocabulary resolution rate
        try:
            vocab_resolved = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'vocab_resolved' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            report['vocabulary'] = {'resolutions': vocab_resolved}
        except Exception:
            report['vocabulary'] = {'resolutions': 0}

        # Heartbeat effectiveness
        try:
            nudges = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'heartbeat_nudge' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            effective = self.conn.execute(
                "SELECT COUNT(*) FROM feature_metrics WHERE metric_type = 'heartbeat_effective' AND timestamp > %s" % cutoff
            ).fetchone()[0]
            report['heartbeat'] = {
                'nudges': nudges,
                'effective': effective,
                'effectiveness_rate': effective / max(nudges, 1),
            }
        except Exception:
            report['heartbeat'] = {'nudges': 0, 'effective': 0, 'effectiveness_rate': 0}

        # Daemon performance
        try:
            speedups = self.conn.execute(
                "SELECT AVG(value), COUNT(*) FROM feature_metrics WHERE metric_type = 'daemon_speedup' AND timestamp > %s" % cutoff
            ).fetchone()
            report['daemon'] = {
                'avg_speedup': round(speedups[0], 1) if speedups[0] else 0,
                'measurements': speedups[1] if speedups[1] else 0,
            }
        except Exception:
            report['daemon'] = {'avg_speedup': 0, 'measurements': 0}

        # Feature usage
        try:
            usage = self.conn.execute(
                "SELECT feature, COUNT(*) FROM feature_metrics WHERE metric_type = 'feature_usage' AND timestamp > %s GROUP BY feature ORDER BY COUNT(*) DESC LIMIT 10" % cutoff
            ).fetchall()
            report['feature_usage'] = {r[0]: r[1] for r in usage}
        except Exception:
            report['feature_usage'] = {}

        report['period_days'] = days
        return report

    def get_consciousness_signal(self) -> Optional[Dict[str, Any]]:
        """
        Generate a consciousness signal about feature effectiveness.
        Returns None if insufficient data, or a signal dict for the brain to surface.
        """
        report = self.get_effectiveness_report(days=7)

        insights = []

        # Low recall hit rate
        recall = report.get('recall', {})
        if recall.get('attempts', 0) > 10 and recall.get('hit_rate', 0) < 0.3:
            insights.append(
                "Recall hit rate is %.0f%% (%d/%d) — embeddings may need retraining or threshold tuning"
                % (recall['hit_rate'] * 100, recall['hits'], recall['attempts'])
            )

        # Heartbeat not effective
        hb = report.get('heartbeat', {})
        if hb.get('nudges', 0) > 5 and hb.get('effectiveness_rate', 0) < 0.2:
            insights.append(
                "Heartbeat nudges are being ignored (%.0f%% effective) — consider adjusting threshold"
                % (hb['effectiveness_rate'] * 100)
            )

        # Low signal engagement
        for sig_type, sig_data in report.get('signals', {}).items():
            if sig_data.get('surfaced', 0) > 5 and sig_data.get('engagement_rate', 0) < 0.1:
                insights.append(
                    "'%s' signals surfaced %d times but rarely acted on — consider deprioritizing"
                    % (sig_type, sig_data['surfaced'])
                )

        if not insights:
            return None

        return {
            'type': 'feature_effectiveness',
            'insights': insights,
            'report_summary': {
                'recall_hit_rate': recall.get('hit_rate', 0),
                'heartbeat_effectiveness': hb.get('effectiveness_rate', 0),
                'total_measurements': sum(
                    v.get('surfaced', 0) + v.get('acted_on', 0)
                    for v in report.get('signals', {}).values()
                ),
            }
        }
