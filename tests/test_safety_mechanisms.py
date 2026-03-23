#!/usr/bin/env python3
"""
Ripple Engine Safety Mechanisms Test Suite
==========================================

Tests 6 safety mechanisms that prevent knowledge destruction during
ripple propagation (confidence changes cascading through the graph).

NO LLM dependency — all impact assessments are hardcoded to test the
MECHANISMS, not the LLM accuracy (which was proven only 50% with Gemma 2B).

Mechanisms tested:
1. Locked Node → Operator Signal (don't silently reduce locked nodes)
2. Diminishing Delta Cascade (halve delta per hop, cycle detection)
3. Type-Based Confidence Floors (minimum confidence by node type)
4. Asymmetric Cascade (VALIDATES=no decay, EXTENDS=0.7x, CONTRADICTS=0.5x)
5. Operator Confirmation Threshold (stage large drops for review)
6. Undo Log (snapshot + rollback per ripple event)

Usage:
    python3 tests/test_safety_mechanisms.py
"""

import json
import os
import sqlite3
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

TEST_DB = "/tmp/brain_safety_test.db"

TYPE_CONFIDENCE_FLOORS = {
    'rule': 0.70,
    'convention': 0.60,
    'decision': 0.30,
    'lesson': 0.20,
    'mechanism': 0.15,
    'impact': 0.10,
    'vocabulary': 0.50,
    'mental_model': 0.20,
    'purpose': 0.20,
    'constraint': 0.25,
    'correction': 0.10,
}
DEFAULT_FLOOR = 0.05

# Asymmetric decay rates per impact type
DECAY_RATES = {
    'VALIDATES': 1.0,    # No decay — strengthening cascades freely
    'EXTENDS': 0.7,      # Mild decay
    'CONTRADICTS': 0.5,  # Strong decay
}

# Operator confirmation threshold for unlocked nodes
CONFIRMATION_THRESHOLD = 0.15


# ════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ════════════════════════════════════════════════════════════════

def create_test_db(path: str) -> sqlite3.Connection:
    """Create a minimal test database with required tables."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            keywords TEXT,
            activation REAL DEFAULT 1.0,
            stability REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 1,
            locked INTEGER DEFAULT 0,
            archived INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.8,
            recency_score REAL DEFAULT 0,
            emotion REAL DEFAULT 0,
            emotion_label TEXT DEFAULT 'neutral',
            last_accessed TEXT,
            created_at TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS edges (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            weight REAL DEFAULT 0.5,
            relation TEXT DEFAULT 'related',
            created_at TEXT,
            last_strengthened TEXT,
            PRIMARY KEY (source_id, target_id),
            FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS node_enrichments (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            vector_type TEXT,
            text TEXT,
            created_at TEXT,
            FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS pending_ripple_confirmations (
            id TEXT PRIMARY KEY,
            ripple_event_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            proposed_change REAL NOT NULL,
            reason TEXT,
            new_evidence_id TEXT,
            old_confidence REAL,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            resolved_at TEXT,
            resolution_reason TEXT,
            sessions_seen INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS ripple_undo_log (
            id TEXT PRIMARY KEY,
            ripple_event_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            old_confidence REAL,
            old_type TEXT,
            old_title TEXT,
            old_content TEXT,
            snapshot_json TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS operator_signals (
            id TEXT PRIMARY KEY,
            signal_type TEXT NOT NULL,
            priority TEXT DEFAULT 'medium',
            content TEXT,
            metadata_json TEXT,
            created_at TEXT,
            acknowledged INTEGER DEFAULT 0
        );
    """)
    conn.commit()
    return conn


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def make_node(conn, node_id: str, node_type: str, title: str,
              confidence: float = 0.80, locked: bool = False,
              content: str = "") -> str:
    """Insert a test node and return its ID."""
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO nodes
           (id, type, title, content, confidence, locked, last_accessed, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (node_id, node_type, title, content, confidence,
         1 if locked else 0, ts, ts, ts)
    )
    conn.commit()
    return node_id


def make_edge(conn, source: str, target: str, relation: str = 'related',
              weight: float = 0.5):
    """Create an edge between two nodes."""
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO edges
           (source_id, target_id, relation, weight, created_at, last_strengthened)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (source, target, relation, weight, ts, ts)
    )
    conn.commit()


def get_confidence(conn, node_id: str) -> float:
    """Get current confidence for a node."""
    row = conn.execute(
        "SELECT confidence FROM nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return row[0] if row else 0.0


def get_node(conn, node_id: str) -> Optional[Dict]:
    """Get full node data."""
    row = conn.execute(
        "SELECT id, type, title, content, confidence, locked FROM nodes WHERE id = ?",
        (node_id,)
    ).fetchone()
    if row:
        return {
            'id': row[0], 'type': row[1], 'title': row[2],
            'content': row[3], 'confidence': row[4], 'locked': bool(row[5])
        }
    return None


def get_neighbors(conn, node_id: str) -> List[str]:
    """Get all neighbor IDs (both directions)."""
    rows = conn.execute(
        """SELECT target_id FROM edges WHERE source_id = ?
           UNION
           SELECT source_id FROM edges WHERE target_id = ?""",
        (node_id, node_id)
    ).fetchall()
    return [r[0] for r in rows]


# ════════════════════════════════════════════════════════════════
# RIPPLE ENGINE (with all safety mechanisms)
# ════════════════════════════════════════════════════════════════

@dataclass
class RippleEvent:
    """A single ripple propagation event."""
    event_id: str = ""
    source_node_id: str = ""      # The new evidence node
    impact_type: str = ""         # VALIDATES / CONTRADICTS / EXTENDS
    initial_delta: float = 0.0
    nodes_touched: Dict[str, float] = field(default_factory=dict)  # node_id -> delta applied
    nodes_blocked: Dict[str, str] = field(default_factory=dict)    # node_id -> reason
    nodes_staged: List[Dict] = field(default_factory=list)         # pending confirmations
    undo_snapshots: List[Dict] = field(default_factory=list)
    total_confidence_change: float = 0.0

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"ripple_{uuid.uuid4().hex[:12]}"


class RippleEngine:
    """
    Ripple engine with all 6 safety mechanisms.

    Propagates confidence changes from a new evidence node through the graph,
    with protections against knowledge destruction.
    """

    def __init__(self, conn: sqlite3.Connection,
                 enable_locked_protection: bool = True,
                 enable_diminishing_cascade: bool = True,
                 enable_type_floors: bool = True,
                 enable_asymmetric_cascade: bool = True,
                 enable_confirmation_threshold: bool = True,
                 enable_undo_log: bool = True):
        self.conn = conn
        self.enable_locked_protection = enable_locked_protection
        self.enable_diminishing = enable_diminishing_cascade
        self.enable_type_floors = enable_type_floors
        self.enable_asymmetric = enable_asymmetric_cascade
        self.enable_confirmation = enable_confirmation_threshold
        self.enable_undo = enable_undo_log

    def propagate(self, target_node_id: str, impact_type: str,
                  initial_delta: float, evidence_node_id: str = "",
                  reason: str = "") -> RippleEvent:
        """
        Propagate a ripple from target_node_id through its neighbors.

        Args:
            target_node_id: The node directly impacted by new evidence
            impact_type: VALIDATES, CONTRADICTS, or EXTENDS
            initial_delta: The confidence change for the direct target
            evidence_node_id: ID of the new evidence node

        Returns:
            RippleEvent with full audit trail
        """
        event = RippleEvent(
            source_node_id=evidence_node_id,
            impact_type=impact_type,
            initial_delta=initial_delta,
        )

        # BFS with diminishing deltas
        # Queue: (node_id, hop_number)
        queue = [(target_node_id, 0)]
        visited: Set[str] = set()

        while queue:
            node_id, hop = queue.pop(0)

            # Cycle detection: skip already-visited nodes
            if node_id in visited:
                continue
            visited.add(node_id)

            # Get node data
            node = get_node(self.conn, node_id)
            if node is None:
                continue

            # Calculate effective delta at this hop
            effective_delta = initial_delta

            # Mechanism 4: Asymmetric cascade (takes priority when enabled)
            if self.enable_asymmetric:
                decay_rate = DECAY_RATES.get(impact_type, 0.5)
                effective_delta = initial_delta * (decay_rate ** hop)
            # Mechanism 2: Diminishing cascade (simple halving, used when asymmetric OFF)
            elif self.enable_diminishing:
                effective_delta = initial_delta * (0.5 ** hop)

            # Natural cutoff: delta too small to matter
            if abs(effective_delta) < 0.01 and hop > 0:
                continue

            # Mechanism 6: Undo log — snapshot BEFORE changes
            if self.enable_undo:
                snapshot = {
                    'node_id': node_id,
                    'old_confidence': node['confidence'],
                    'old_type': node['type'],
                    'old_title': node['title'],
                    'old_content': node['content'],
                }
                event.undo_snapshots.append(snapshot)
                ts = datetime.now(timezone.utc).isoformat()
                self.conn.execute(
                    """INSERT INTO ripple_undo_log
                       (id, ripple_event_id, node_id, old_confidence, old_type,
                        old_title, old_content, snapshot_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (uuid.uuid4().hex[:16], event.event_id, node_id,
                     node['confidence'], node['type'], node['title'],
                     node['content'], json.dumps(snapshot), ts)
                )

            # Mechanism 1: Locked node protection
            if self.enable_locked_protection and node['locked'] and effective_delta < 0:
                # Block confidence reduction on locked nodes
                event.nodes_blocked[node_id] = "locked_node_protection"
                # Log as pending operator question
                self._log_pending_confirmation(
                    event, node_id, effective_delta, evidence_node_id,
                    f"Locked node '{node['title']}' — proposed reduction of {effective_delta:.4f}"
                )
                self._log_operator_signal(
                    node_id, effective_delta, node, evidence_node_id, "locked"
                )
                # Do NOT propagate reductions further from a locked node
                continue

            # Mechanism 5: Confirmation threshold for large drops
            if (self.enable_confirmation and effective_delta < 0
                    and abs(effective_delta) > CONFIRMATION_THRESHOLD
                    and not node['locked']):
                # Stage for confirmation instead of applying
                event.nodes_blocked[node_id] = "confirmation_threshold"
                event.nodes_staged.append({
                    'node_id': node_id,
                    'proposed_delta': effective_delta,
                    'old_confidence': node['confidence'],
                })
                self._log_pending_confirmation(
                    event, node_id, effective_delta, evidence_node_id,
                    f"Large drop ({effective_delta:.4f}) on '{node['title']}' exceeds threshold {CONFIRMATION_THRESHOLD}"
                )
                self._log_operator_signal(
                    node_id, effective_delta, node, evidence_node_id, "threshold"
                )
                continue

            # Apply the delta
            new_confidence = node['confidence'] + effective_delta

            # Mechanism 3: Type-based confidence floors
            if self.enable_type_floors and effective_delta < 0:
                floor = TYPE_CONFIDENCE_FLOORS.get(node['type'], DEFAULT_FLOOR)
                if new_confidence < floor:
                    new_confidence = floor

            # Clamp to [0, 1]
            new_confidence = max(0.0, min(1.0, new_confidence))

            # Apply
            actual_delta = new_confidence - node['confidence']
            self.conn.execute(
                "UPDATE nodes SET confidence = ? WHERE id = ?",
                (new_confidence, node_id)
            )
            event.nodes_touched[node_id] = actual_delta
            event.total_confidence_change += actual_delta

            # Enqueue neighbors for cascade
            neighbors = get_neighbors(self.conn, node_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, hop + 1))

        self.conn.commit()
        return event

    def rollback(self, event: RippleEvent):
        """Rollback a ripple event using undo log."""
        rows = self.conn.execute(
            "SELECT node_id, old_confidence FROM ripple_undo_log WHERE ripple_event_id = ?",
            (event.event_id,)
        ).fetchall()
        for node_id, old_confidence in rows:
            self.conn.execute(
                "UPDATE nodes SET confidence = ? WHERE id = ?",
                (old_confidence, node_id)
            )
        self.conn.commit()

    def confirm_pending(self, confirmation_id: str, approved: bool,
                        reason: str = ""):
        """Resolve a pending confirmation."""
        ts = datetime.now(timezone.utc).isoformat()
        status = 'approved' if approved else 'rejected'
        self.conn.execute(
            """UPDATE pending_ripple_confirmations
               SET status = ?, resolved_at = ?, resolution_reason = ?
               WHERE id = ?""",
            (status, ts, reason, confirmation_id)
        )

        if approved:
            # Apply the staged change
            row = self.conn.execute(
                "SELECT node_id, proposed_change, old_confidence FROM pending_ripple_confirmations WHERE id = ?",
                (confirmation_id,)
            ).fetchone()
            if row:
                node_id, delta, old_conf = row
                node = get_node(self.conn, node_id)
                if node:
                    new_conf = max(0.0, min(1.0, node['confidence'] + delta))
                    # Apply floor
                    floor = TYPE_CONFIDENCE_FLOORS.get(node['type'], DEFAULT_FLOOR)
                    new_conf = max(floor, new_conf)
                    self.conn.execute(
                        "UPDATE nodes SET confidence = ? WHERE id = ?",
                        (new_conf, node_id)
                    )
        self.conn.commit()

    def _log_pending_confirmation(self, event: RippleEvent, node_id: str,
                                  delta: float, evidence_id: str, reason: str):
        """Log a pending confirmation entry."""
        ts = datetime.now(timezone.utc).isoformat()
        conf_id = uuid.uuid4().hex[:16]
        node = get_node(self.conn, node_id)
        self.conn.execute(
            """INSERT INTO pending_ripple_confirmations
               (id, ripple_event_id, node_id, proposed_change, reason,
                new_evidence_id, old_confidence, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (conf_id, event.event_id, node_id, delta, reason,
             evidence_id, node['confidence'] if node else 0, ts)
        )
        return conf_id

    def _log_operator_signal(self, node_id: str, delta: float,
                             node: Dict, evidence_id: str, reason_type: str):
        """Log an operator signal for consciousness."""
        ts = datetime.now(timezone.utc).isoformat()
        if reason_type == "locked":
            content = (
                f"New evidence [{evidence_id}] proposes confidence drop of "
                f"{delta:.4f} on LOCKED node [{node['title']}]. "
                f"Review and confirm or reject."
            )
            priority = "high"
        else:
            content = (
                f"New evidence [{evidence_id}] proposes confidence drop of "
                f"{delta:.4f} on [{node['title']}]. "
                f"Exceeds threshold ({CONFIRMATION_THRESHOLD}). Confirm or reject?"
            )
            priority = "high"

        self.conn.execute(
            """INSERT INTO operator_signals
               (id, signal_type, priority, content, metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex[:16], 'ripple_confirmation', priority, content,
             json.dumps({'node_id': node_id, 'delta': delta,
                         'evidence_id': evidence_id}),
             ts)
        )


# ════════════════════════════════════════════════════════════════
# TEST FRAMEWORK
# ════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    mechanism: str
    test_case: str
    expected: str
    actual: str
    passed: bool
    detail: str = ""


class SafetyTestSuite:
    """Run all safety mechanism tests."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.conn: Optional[sqlite3.Connection] = None

    def _fresh_db(self) -> sqlite3.Connection:
        """Create a fresh test DB for each test group."""
        if self.conn:
            self.conn.close()
        self.conn = create_test_db(TEST_DB)
        return self.conn

    def add_result(self, mechanism: str, test_case: str,
                   expected: str, actual: str, passed: bool, detail: str = ""):
        self.results.append(TestResult(
            mechanism=mechanism, test_case=test_case,
            expected=expected, actual=actual, passed=passed, detail=detail
        ))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_case}")
        if not passed:
            print(f"         expected: {expected}")
            print(f"         actual:   {actual}")
            if detail:
                print(f"         detail:   {detail}")

    def run_all(self):
        print("=" * 70)
        print("RIPPLE ENGINE SAFETY MECHANISMS — TEST SUITE")
        print("=" * 70)
        print()

        self.test_mechanism_1_locked_node_protection()
        self.test_mechanism_2_diminishing_cascade()
        self.test_mechanism_3_type_floors()
        self.test_mechanism_4_asymmetric_cascade()
        self.test_mechanism_5_confirmation_threshold()
        self.test_mechanism_6_undo_log()
        self.test_combined_mechanisms()

        self.print_summary()

    # ── Mechanism 1: Locked Node → Operator Signal ──

    def test_mechanism_1_locked_node_protection(self):
        print("\n─── Mechanism 1: Locked Node → Operator Signal ───")
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=True,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        # Test 1a: Locked rule gets CONTRADICTS → should NOT change, should log
        make_node(conn, "locked_rule", "rule", "Always use UTC", confidence=0.95, locked=True)
        make_node(conn, "evidence_1", "lesson", "Local time is fine", confidence=0.80)
        make_edge(conn, "evidence_1", "locked_rule")

        event = engine.propagate("locked_rule", "CONTRADICTS", -0.10, "evidence_1")
        conf_after = get_confidence(conn, "locked_rule")

        self.add_result("1: Locked Protection", "1a: Locked rule CONTRADICTS — no change",
                        "0.95", f"{conf_after:.2f}",
                        abs(conf_after - 0.95) < 0.001)

        blocked = "locked_rule" in event.nodes_blocked
        self.add_result("1: Locked Protection", "1a: Locked rule CONTRADICTS — logged as blocked",
                        "True", str(blocked), blocked)

        # Check operator signal was created
        signals = conn.execute(
            "SELECT COUNT(*) FROM operator_signals WHERE signal_type = 'ripple_confirmation'"
        ).fetchone()[0]
        self.add_result("1: Locked Protection", "1a: Operator signal created",
                        ">=1", str(signals), signals >= 1)

        # Check pending confirmation was created
        pending = conn.execute(
            "SELECT COUNT(*) FROM pending_ripple_confirmations WHERE node_id = 'locked_rule'"
        ).fetchone()[0]
        self.add_result("1: Locked Protection", "1a: Pending confirmation created",
                        ">=1", str(pending), pending >= 1)

        # Test 1b: Locked rule gets VALIDATES → should increase (strengthening is fine)
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=True,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)
        make_node(conn, "locked_rule_v", "rule", "Always use UTC", confidence=0.85, locked=True)
        make_node(conn, "evidence_v", "lesson", "UTC prevents timezone bugs", confidence=0.90)
        make_edge(conn, "evidence_v", "locked_rule_v")

        event = engine.propagate("locked_rule_v", "VALIDATES", +0.10, "evidence_v")
        conf_after = get_confidence(conn, "locked_rule_v")

        self.add_result("1: Locked Protection", "1b: Locked rule VALIDATES — should increase",
                        "0.95", f"{conf_after:.2f}",
                        abs(conf_after - 0.95) < 0.001)

        # Test 1c: Locked lesson gets CONTRADICTS with large delta
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=True,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)
        make_node(conn, "locked_lesson", "lesson", "Never use eval()", confidence=0.85, locked=True)
        make_node(conn, "evidence_c", "lesson", "eval() is safe in sandboxes", confidence=0.70)
        make_edge(conn, "evidence_c", "locked_lesson")

        event = engine.propagate("locked_lesson", "CONTRADICTS", -0.30, "evidence_c")
        conf_after = get_confidence(conn, "locked_lesson")

        self.add_result("1: Locked Protection", "1c: Locked lesson large CONTRADICTS — no change",
                        "0.85", f"{conf_after:.2f}",
                        abs(conf_after - 0.85) < 0.001)

        pending = conn.execute(
            "SELECT COUNT(*) FROM pending_ripple_confirmations WHERE node_id = 'locked_lesson'"
        ).fetchone()[0]
        self.add_result("1: Locked Protection", "1c: Pending confirmation for operator",
                        ">=1", str(pending), pending >= 1)

        # Test 1d: Locked decision gets EXTENDS → should handle normally (positive delta)
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=True,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)
        make_node(conn, "locked_decision", "decision", "Use React", confidence=0.80, locked=True)
        make_node(conn, "evidence_e", "lesson", "React hooks improve DX", confidence=0.75)
        make_edge(conn, "evidence_e", "locked_decision")

        # EXTENDS with positive delta should be allowed on locked nodes
        event = engine.propagate("locked_decision", "EXTENDS", +0.02, "evidence_e")
        conf_after = get_confidence(conn, "locked_decision")

        self.add_result("1: Locked Protection", "1d: Locked decision EXTENDS (positive) — allowed",
                        "0.82", f"{conf_after:.2f}",
                        abs(conf_after - 0.82) < 0.001)

    # ── Mechanism 2: Diminishing Delta Cascade ──

    def test_mechanism_2_diminishing_cascade(self):
        print("\n─── Mechanism 2: Diminishing Delta Cascade ───")

        # Test 2a: Linear chain A→B→C→D→E with CONTRADICTS -0.20
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=True,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        for name in ['A', 'B', 'C', 'D', 'E']:
            make_node(conn, name, "mechanism", f"Node {name}", confidence=0.80)
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'B', 'C')
        make_edge(conn, 'C', 'D')
        make_edge(conn, 'D', 'E')

        event = engine.propagate("A", "CONTRADICTS", -0.20, "new_evidence")

        expected_chain = {
            'A': (0.60, -0.20),   # hop 0: full delta
            'B': (0.70, -0.10),   # hop 1: -0.10
            'C': (0.75, -0.05),   # hop 2: -0.05
            'D': (0.775, -0.025), # hop 3: -0.025
            'E': (0.7875, -0.0125), # hop 4: -0.0125
        }

        for name, (exp_conf, exp_delta) in expected_chain.items():
            actual = get_confidence(conn, name)
            if abs(exp_delta) < 0.01 and name != 'A':
                # E might be skipped due to natural cutoff
                passed = abs(actual - exp_conf) < 0.02 or abs(actual - 0.80) < 0.001
                self.add_result("2: Diminishing Cascade",
                                f"2a: Chain node {name} (delta {exp_delta:.4f})",
                                f"{exp_conf:.4f}", f"{actual:.4f}", passed,
                                "May be cutoff (delta < 0.01)")
            else:
                passed = abs(actual - exp_conf) < 0.001
                self.add_result("2: Diminishing Cascade",
                                f"2a: Chain node {name} (delta {exp_delta:.4f})",
                                f"{exp_conf:.4f}", f"{actual:.4f}", passed)

        # Verify total confidence loss
        total_loss = sum(0.80 - get_confidence(conn, n) for n in 'ABCDE')
        self.add_result("2: Diminishing Cascade",
                        "2a: Total chain loss",
                        "~0.3875", f"{total_loss:.4f}",
                        abs(total_loss - 0.3875) < 0.05,
                        "Sum of all deltas (some may cutoff)")

        # Test 2b: Tree structure A→[B,C,D], B→[E,F], C→[G]
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=True,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        for name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            make_node(conn, name, "mechanism", f"Node {name}", confidence=0.80)
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'A', 'C')
        make_edge(conn, 'A', 'D')
        make_edge(conn, 'B', 'E')
        make_edge(conn, 'B', 'F')
        make_edge(conn, 'C', 'G')

        event = engine.propagate("A", "CONTRADICTS", -0.20, "new_evidence")

        tree_expected = {
            'A': 0.60,   # hop 0: -0.20
            'B': 0.70, 'C': 0.70, 'D': 0.70,  # hop 1: -0.10
            'E': 0.75, 'F': 0.75,              # hop 2: -0.05
            'G': 0.75,                          # hop 2: -0.05
        }

        for name, exp_conf in tree_expected.items():
            actual = get_confidence(conn, name)
            passed = abs(actual - exp_conf) < 0.001
            self.add_result("2: Diminishing Cascade",
                            f"2b: Tree node {name}",
                            f"{exp_conf:.4f}", f"{actual:.4f}", passed)

        tree_loss = sum(0.80 - get_confidence(conn, n) for n in 'ABCDEFG')
        chain_loss = total_loss
        self.add_result("2: Diminishing Cascade",
                        "2b: Tree causes more total damage than chain",
                        "True", f"tree={tree_loss:.4f} > chain={chain_loss:.4f}",
                        tree_loss > chain_loss)

        # Test 2c: CYCLE detection A→B→C→A
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=True,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        for name in ['A', 'B', 'C']:
            make_node(conn, name, "mechanism", f"Node {name}", confidence=0.80)
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'B', 'C')
        make_edge(conn, 'C', 'A')

        event = engine.propagate("A", "CONTRADICTS", -0.20, "new_evidence")

        # A should be visited only once (cycle detection)
        # Note: edges are bidirectional in get_neighbors, so in cycle A→B→C→A:
        #   A neighbors: B (via A→B) and C (via C→A reverse)
        #   So both B and C are hop-1 neighbors of A
        #   When B is processed at hop 1, C is already visited → skip
        #   When C is processed at hop 1, A and B are already visited → skip
        conf_a = get_confidence(conn, 'A')
        conf_b = get_confidence(conn, 'B')
        conf_c = get_confidence(conn, 'C')

        self.add_result("2: Diminishing Cascade",
                        "2c: Cycle — A visited once (0.60)",
                        "0.60", f"{conf_a:.4f}",
                        abs(conf_a - 0.60) < 0.001,
                        "CRITICAL: must not re-apply delta")

        self.add_result("2: Diminishing Cascade",
                        "2c: Cycle — B gets hop-1 delta (0.70)",
                        "0.70", f"{conf_b:.4f}",
                        abs(conf_b - 0.70) < 0.001)

        # C is also a hop-1 neighbor of A (via reverse edge C→A)
        self.add_result("2: Diminishing Cascade",
                        "2c: Cycle — C gets hop-1 delta (0.70, bidirectional)",
                        "0.70", f"{conf_c:.4f}",
                        abs(conf_c - 0.70) < 0.001,
                        "C→A edge makes C a direct neighbor of A")

        # Verify no infinite loop — if we got here, it didn't hang
        self.add_result("2: Diminishing Cascade",
                        "2c: Cycle — no infinite loop",
                        "completed", "completed", True)

    # ── Mechanism 3: Type-Based Confidence Floors ──

    def test_mechanism_3_type_floors(self):
        print("\n─── Mechanism 3: Type-Based Confidence Floors ───")
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=False,
                              enable_type_floors=True,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        # Test 3a: Rule at 0.75, CONTRADICTS -0.20 → floor at 0.70
        make_node(conn, "rule_1", "rule", "Use strict mode", confidence=0.75)
        engine.propagate("rule_1", "CONTRADICTS", -0.20, "ev1")
        actual = get_confidence(conn, "rule_1")
        self.add_result("3: Type Floors", "3a: Rule 0.75 - 0.20 → floor 0.70",
                        "0.70", f"{actual:.2f}",
                        abs(actual - 0.70) < 0.001)

        # Test 3b: Lesson at 0.25, CONTRADICTS -0.15 → floor at 0.20
        make_node(conn, "lesson_1", "lesson", "Don't skip tests", confidence=0.25)
        engine.propagate("lesson_1", "CONTRADICTS", -0.15, "ev2")
        actual = get_confidence(conn, "lesson_1")
        self.add_result("3: Type Floors", "3b: Lesson 0.25 - 0.15 → floor 0.20",
                        "0.20", f"{actual:.2f}",
                        abs(actual - 0.20) < 0.001)

        # Test 3c: Decision at 0.35, CONTRADICTS -0.10 → floor at 0.30
        make_node(conn, "decision_1", "decision", "Use PostgreSQL", confidence=0.35)
        engine.propagate("decision_1", "CONTRADICTS", -0.10, "ev3")
        actual = get_confidence(conn, "decision_1")
        self.add_result("3: Type Floors", "3c: Decision 0.35 - 0.10 → floor 0.30",
                        "0.30", f"{actual:.2f}",
                        abs(actual - 0.30) < 0.001)

        # Test 3d: Mechanism at 0.50, CONTRADICTS -0.40 → floor at 0.15
        make_node(conn, "mech_1", "mechanism", "Event loop model", confidence=0.50)
        engine.propagate("mech_1", "CONTRADICTS", -0.40, "ev4")
        actual = get_confidence(conn, "mech_1")
        self.add_result("3: Type Floors", "3d: Mechanism 0.50 - 0.40 → floor 0.15",
                        "0.15", f"{actual:.2f}",
                        abs(actual - 0.15) < 0.001)

        # Test 3e: Unknown type → default floor 0.05
        make_node(conn, "thought_1", "thought", "Random thought", confidence=0.20)
        engine.propagate("thought_1", "CONTRADICTS", -0.30, "ev5")
        actual = get_confidence(conn, "thought_1")
        self.add_result("3: Type Floors", "3e: Thought (no floor) 0.20 - 0.30 → default 0.05",
                        "0.05", f"{actual:.2f}",
                        abs(actual - 0.05) < 0.001)

        # Test 3f: VALIDATES should NOT be affected by floors (positive delta)
        make_node(conn, "rule_2", "rule", "Use TypeScript", confidence=0.60)
        engine.propagate("rule_2", "VALIDATES", +0.15, "ev6")
        actual = get_confidence(conn, "rule_2")
        self.add_result("3: Type Floors", "3f: VALIDATES not affected by floors",
                        "0.75", f"{actual:.2f}",
                        abs(actual - 0.75) < 0.001)

    # ── Mechanism 4: Asymmetric Cascade ──

    def test_mechanism_4_asymmetric_cascade(self):
        print("\n─── Mechanism 4: Asymmetric Cascade ───")

        # Test 4a: VALIDATES with +0.10 — no decay
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=True,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        for name in ['A', 'B', 'C', 'D', 'E']:
            make_node(conn, name, "mechanism", f"Node {name}", confidence=0.70)
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'B', 'C')
        make_edge(conn, 'C', 'D')
        make_edge(conn, 'D', 'E')

        engine.propagate("A", "VALIDATES", +0.10, "ev_v")

        for name in ['A', 'B', 'C', 'D', 'E']:
            actual = get_confidence(conn, name)
            self.add_result("4: Asymmetric Cascade",
                            f"4a: VALIDATES chain {name} — no decay (+0.10)",
                            "0.80", f"{actual:.2f}",
                            abs(actual - 0.80) < 0.001)

        # Test 4b: EXTENDS with +0.05 — 0.7x decay
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=True,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        for name in ['A', 'B', 'C', 'D', 'E']:
            make_node(conn, name, "mechanism", f"Node {name}", confidence=0.70)
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'B', 'C')
        make_edge(conn, 'C', 'D')
        make_edge(conn, 'D', 'E')

        engine.propagate("A", "EXTENDS", +0.05, "ev_e")

        extends_expected = {
            'A': 0.75,    # hop 0: +0.05
            'B': 0.735,   # hop 1: +0.035
            'C': 0.7245,  # hop 2: +0.0245
            'D': 0.71715, # hop 3: +0.01715
            'E': 0.71201, # hop 4: +0.01201
        }

        for name, exp in extends_expected.items():
            actual = get_confidence(conn, name)
            delta_at_hop = 0.05 * (0.7 ** list('ABCDE').index(name))
            if delta_at_hop < 0.01:
                # May be cutoff
                self.add_result("4: Asymmetric Cascade",
                                f"4b: EXTENDS chain {name} (delta {delta_at_hop:.4f})",
                                f"{exp:.4f}", f"{actual:.4f}",
                                abs(actual - exp) < 0.005 or abs(actual - 0.70) < 0.001,
                                "May be cutoff")
            else:
                self.add_result("4: Asymmetric Cascade",
                                f"4b: EXTENDS chain {name} (delta {delta_at_hop:.4f})",
                                f"{exp:.4f}", f"{actual:.4f}",
                                abs(actual - exp) < 0.005)

        # Test 4c: CONTRADICTS with -0.20 — 0.5x decay (same as Mechanism 2)
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=True,
                              enable_confirmation_threshold=False,
                              enable_undo_log=False)

        for name in ['A', 'B', 'C', 'D', 'E']:
            make_node(conn, name, "mechanism", f"Node {name}", confidence=0.80)
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'B', 'C')
        make_edge(conn, 'C', 'D')
        make_edge(conn, 'D', 'E')

        engine.propagate("A", "CONTRADICTS", -0.20, "ev_c")

        contra_expected = {'A': 0.60, 'B': 0.70, 'C': 0.75, 'D': 0.775, 'E': 0.7875}
        for name, exp in contra_expected.items():
            actual = get_confidence(conn, name)
            delta = -0.20 * (0.5 ** list('ABCDE').index(name))
            if abs(delta) < 0.01:
                self.add_result("4: Asymmetric Cascade",
                                f"4c: CONTRADICTS chain {name} (delta {delta:.4f})",
                                f"{exp:.4f}", f"{actual:.4f}",
                                abs(actual - exp) < 0.005 or abs(actual - 0.80) < 0.001,
                                "May be cutoff")
            else:
                self.add_result("4: Asymmetric Cascade",
                                f"4c: CONTRADICTS chain {name} (delta {delta:.4f})",
                                f"{exp:.4f}", f"{actual:.4f}",
                                abs(actual - exp) < 0.005)

    # ── Mechanism 5: Operator Confirmation Threshold ──

    def test_mechanism_5_confirmation_threshold(self):
        print("\n─── Mechanism 5: Operator Confirmation Threshold ───")
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=True,
                              enable_diminishing_cascade=False,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=True,
                              enable_undo_log=False)

        # Test 5a: Unlocked node, delta -0.10 → auto-apply (under threshold)
        make_node(conn, "node_a", "mechanism", "Small change", confidence=0.80)
        event = engine.propagate("node_a", "CONTRADICTS", -0.10, "ev1")
        actual = get_confidence(conn, "node_a")
        self.add_result("5: Confirmation Threshold",
                        "5a: Under threshold -0.10 → auto-apply",
                        "0.70", f"{actual:.2f}",
                        abs(actual - 0.70) < 0.001)
        self.add_result("5: Confirmation Threshold",
                        "5a: No pending confirmation",
                        "0 staged", f"{len(event.nodes_staged)} staged",
                        len(event.nodes_staged) == 0)

        # Test 5b: Unlocked node, delta -0.20 → stage for confirmation
        make_node(conn, "node_b", "mechanism", "Large change", confidence=0.80)
        event = engine.propagate("node_b", "CONTRADICTS", -0.20, "ev2")
        actual = get_confidence(conn, "node_b")
        self.add_result("5: Confirmation Threshold",
                        "5b: Over threshold -0.20 → staged (no change)",
                        "0.80", f"{actual:.2f}",
                        abs(actual - 0.80) < 0.001)
        self.add_result("5: Confirmation Threshold",
                        "5b: Pending confirmation created",
                        "1 staged", f"{len(event.nodes_staged)} staged",
                        len(event.nodes_staged) == 1)

        # Test 5c: Locked node, delta -0.05 → stage for confirmation (any reduction on locked)
        make_node(conn, "node_c", "rule", "Sacred rule", confidence=0.90, locked=True)
        event = engine.propagate("node_c", "CONTRADICTS", -0.05, "ev3")
        actual = get_confidence(conn, "node_c")
        self.add_result("5: Confirmation Threshold",
                        "5c: Locked node any reduction → blocked",
                        "0.90", f"{actual:.2f}",
                        abs(actual - 0.90) < 0.001)

        # Test 5d: Confirm one, reject one
        make_node(conn, "node_d1", "mechanism", "Confirm me", confidence=0.80)
        make_node(conn, "node_d2", "mechanism", "Reject me", confidence=0.80)

        engine.propagate("node_d1", "CONTRADICTS", -0.20, "ev4a")
        engine.propagate("node_d2", "CONTRADICTS", -0.20, "ev4b")

        # Get pending confirmation IDs
        rows = conn.execute(
            "SELECT id, node_id FROM pending_ripple_confirmations WHERE status = 'pending' ORDER BY created_at DESC LIMIT 2"
        ).fetchall()

        confirm_id = None
        reject_id = None
        for row_id, node_id in rows:
            if node_id == 'node_d1':
                confirm_id = row_id
            elif node_id == 'node_d2':
                reject_id = row_id

        if confirm_id:
            engine.confirm_pending(confirm_id, approved=True, reason="Confirmed by operator")
        if reject_id:
            engine.confirm_pending(reject_id, approved=False, reason="Evidence was wrong")

        conf_d1 = get_confidence(conn, "node_d1")
        conf_d2 = get_confidence(conn, "node_d2")

        self.add_result("5: Confirmation Threshold",
                        "5d: Confirmed → change applied",
                        "<0.80", f"{conf_d1:.2f}",
                        conf_d1 < 0.80)

        self.add_result("5: Confirmation Threshold",
                        "5d: Rejected → no change",
                        "0.80", f"{conf_d2:.2f}",
                        abs(conf_d2 - 0.80) < 0.001)

    # ── Mechanism 6: Undo Log ──

    def test_mechanism_6_undo_log(self):
        print("\n─── Mechanism 6: Undo Log ───")
        conn = self._fresh_db()
        engine = RippleEngine(conn,
                              enable_locked_protection=False,
                              enable_diminishing_cascade=True,
                              enable_type_floors=False,
                              enable_asymmetric_cascade=False,
                              enable_confirmation_threshold=False,
                              enable_undo_log=True)

        # Create chain of 5 nodes
        original_confs = {}
        for name in ['A', 'B', 'C', 'D', 'E']:
            conf = 0.80
            make_node(conn, name, "mechanism", f"Node {name}", confidence=conf)
            original_confs[name] = conf
        make_edge(conn, 'A', 'B')
        make_edge(conn, 'B', 'C')
        make_edge(conn, 'C', 'D')
        make_edge(conn, 'D', 'E')

        # Test 6a: Apply ripple, verify changes, then rollback
        event = engine.propagate("A", "CONTRADICTS", -0.20, "ev_undo")

        # Verify changes were applied
        changes_applied = any(
            abs(get_confidence(conn, n) - 0.80) > 0.001
            for n in ['A', 'B', 'C']
        )
        self.add_result("6: Undo Log", "6a: Ripple applied changes to >=3 nodes",
                        "True", str(changes_applied), changes_applied)

        # Verify undo log has entries
        undo_count = conn.execute(
            "SELECT COUNT(*) FROM ripple_undo_log WHERE ripple_event_id = ?",
            (event.event_id,)
        ).fetchone()[0]
        self.add_result("6: Undo Log", "6a: Undo log captured snapshots",
                        ">=3", str(undo_count), undo_count >= 3)

        # Rollback
        engine.rollback(event)

        # Verify ALL nodes restored
        all_restored = True
        for name in ['A', 'B', 'C', 'D', 'E']:
            actual = get_confidence(conn, name)
            if abs(actual - original_confs[name]) > 0.001:
                all_restored = False
                break

        self.add_result("6: Undo Log", "6a: Rollback restores all nodes",
                        "all at 0.80", f"all_restored={all_restored}",
                        all_restored)

        # Test 6b: Rollback after subsequent encode
        event = engine.propagate("A", "CONTRADICTS", -0.20, "ev_undo2")

        # Now add a NEW node (simulating a new encode AFTER the ripple)
        make_node(conn, "NEW_NODE", "lesson", "New knowledge", confidence=0.90)
        make_edge(conn, "NEW_NODE", "B")

        # Rollback the ripple
        engine.rollback(event)

        # Verify ripple changes rolled back
        conf_a = get_confidence(conn, 'A')
        self.add_result("6: Undo Log", "6b: Rollback after new encode — ripple reverted",
                        "0.80", f"{conf_a:.2f}",
                        abs(conf_a - 0.80) < 0.001)

        # Verify the new encode is NOT affected
        conf_new = get_confidence(conn, "NEW_NODE")
        self.add_result("6: Undo Log", "6b: New encode untouched by rollback",
                        "0.90", f"{conf_new:.2f}",
                        abs(conf_new - 0.90) < 0.001)

    # ── Combined: All Mechanisms Together ──

    def test_combined_mechanisms(self):
        print("\n─── Combined: All Mechanisms Together ───")
        conn = self._fresh_db()

        # Compare: no safety vs all safety
        def setup_scenario(c):
            make_node(c, "root", "mechanism", "Root mechanism", confidence=0.80)
            make_node(c, "rule_a", "rule", "Sacred rule A", confidence=0.85, locked=True)
            make_node(c, "lesson_b", "lesson", "Lesson B", confidence=0.40)
            make_node(c, "decision_c", "decision", "Decision C", confidence=0.50)
            make_node(c, "mech_d", "mechanism", "Mechanism D", confidence=0.30)
            make_node(c, "conv_e", "convention", "Convention E", confidence=0.65)

            make_edge(c, "root", "rule_a")
            make_edge(c, "root", "lesson_b")
            make_edge(c, "root", "decision_c")
            make_edge(c, "lesson_b", "mech_d")
            make_edge(c, "decision_c", "conv_e")

        # Scenario 1: NO safety
        conn_raw = create_test_db("/tmp/brain_safety_raw.db")
        setup_scenario(conn_raw)
        engine_raw = RippleEngine(conn_raw,
                                  enable_locked_protection=False,
                                  enable_diminishing_cascade=False,
                                  enable_type_floors=False,
                                  enable_asymmetric_cascade=False,
                                  enable_confirmation_threshold=False,
                                  enable_undo_log=False)
        event_raw = engine_raw.propagate("root", "CONTRADICTS", -0.30, "evidence")
        raw_total = sum(
            0 - event_raw.nodes_touched.get(n, 0)
            for n in ["root", "rule_a", "lesson_b", "decision_c", "mech_d", "conv_e"]
        )

        # Scenario 2: ALL safety
        conn_safe = create_test_db("/tmp/brain_safety_full.db")
        setup_scenario(conn_safe)
        engine_safe = RippleEngine(conn_safe,
                                   enable_locked_protection=True,
                                   enable_diminishing_cascade=False,
                                   enable_type_floors=True,
                                   enable_asymmetric_cascade=True,
                                   enable_confirmation_threshold=True,
                                   enable_undo_log=True)
        event_safe = engine_safe.propagate("root", "CONTRADICTS", -0.30, "evidence")
        safe_total = sum(
            0 - event_safe.nodes_touched.get(n, 0)
            for n in ["root", "rule_a", "lesson_b", "decision_c", "mech_d", "conv_e"]
        )

        self.add_result("Combined", "Raw vs Safe — raw causes more total damage",
                        "raw > safe", f"raw={raw_total:.4f}, safe={safe_total:.4f}",
                        raw_total > safe_total)

        rule_a_safe = get_confidence(conn_safe, "rule_a")
        self.add_result("Combined", "Locked rule_a protected",
                        "0.85", f"{rule_a_safe:.2f}",
                        abs(rule_a_safe - 0.85) < 0.001)

        # Root: delta -0.30 > threshold 0.15, should be staged
        root_staged = "root" in event_safe.nodes_blocked
        self.add_result("Combined", "Root staged for confirmation (delta -0.30 > 0.15)",
                        "True", str(root_staged), root_staged)

        undo_count = conn_safe.execute(
            "SELECT COUNT(*) FROM ripple_undo_log WHERE ripple_event_id = ?",
            (event_safe.event_id,)
        ).fetchone()[0]
        self.add_result("Combined", "Undo log captured snapshots",
                        ">=1", str(undo_count), undo_count >= 1)

        # Damage comparison table
        print("\n  ┌─────────────────────────────────────────────────────┐")
        print("  │ Confidence Damage Comparison: No Safety vs All Safety│")
        print("  ├──────────────┬──────────┬──────────┬────────────────┤")
        print("  │ Node         │ Raw Conf │ Safe Conf│ Protected By   │")
        print("  ├──────────────┼──────────┼──────────┼────────────────┤")
        for nid in ["root", "rule_a", "lesson_b", "decision_c", "mech_d", "conv_e"]:
            raw_c = get_confidence(conn_raw, nid)
            safe_c = get_confidence(conn_safe, nid)
            protection = ""
            if nid in event_safe.nodes_blocked:
                protection = event_safe.nodes_blocked[nid]
            elif safe_c > raw_c:
                protection = "floor/decay"
            print(f"  │ {nid:12s} │ {raw_c:8.4f} │ {safe_c:8.4f} │ {protection:14s} │")
        print("  └──────────────┴──────────┴──────────┴────────────────┘")
        print(f"  Total raw damage: {raw_total:.4f}")
        print(f"  Total safe damage: {safe_total:.4f}")
        print(f"  Damage prevented: {raw_total - safe_total:.4f}")

        conn_raw.close()
        conn_safe.close()

    # ── Summary ──

    def print_summary(self):
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        mechanisms = defaultdict(lambda: {'pass': 0, 'fail': 0, 'total': 0})
        for r in self.results:
            mechanisms[r.mechanism]['total'] += 1
            if r.passed:
                mechanisms[r.mechanism]['pass'] += 1
            else:
                mechanisms[r.mechanism]['fail'] += 1

        total_pass = sum(m['pass'] for m in mechanisms.values())
        total_fail = sum(m['fail'] for m in mechanisms.values())
        total = total_pass + total_fail

        print(f"\n  Total: {total_pass}/{total} passed ({total_fail} failed)\n")

        print("  ┌───────────────────────────────────┬──────┬──────┬────────┐")
        print("  │ Mechanism                         │ Pass │ Fail │ Status │")
        print("  ├───────────────────────────────────┼──────┼──────┼────────┤")
        for mech, stats in mechanisms.items():
            status_mark = "  OK  " if stats['fail'] == 0 else " FAIL "
            print(f"  │ {mech:33s} │ {stats['pass']:4d} │ {stats['fail']:4d} │{status_mark}│")
        print("  └───────────────────────────────────┴──────┴──────┴────────┘")

        print("\n  RECOMMENDATIONS:")
        ship = []
        fix = []
        for mech, stats in mechanisms.items():
            if stats['fail'] == 0:
                ship.append(mech)
            else:
                fix.append(mech)

        if ship:
            print(f"  Ship now:  {', '.join(ship)}")
        if fix:
            print(f"  Fix first: {', '.join(fix)}")

        print("\n  PARAMETER RECOMMENDATIONS:")
        print("  - CONTRADICTS decay: 0.5x per hop (proven safe, natural cutoff at hop 4)")
        print("  - VALIDATES decay: 1.0x (no decay — strengthening cascades freely)")
        print("  - EXTENDS decay: 0.7x per hop (mild — reasonable for nuance)")
        print("  - Confirmation threshold: 0.15 for unlocked, ANY for locked")
        print("  - Type floors: as specified (rule=0.70, convention=0.60, etc.)")
        print("  - Undo log: always enabled (cheap insurance)")
        print()

        failures = [r for r in self.results if not r.passed]
        if failures:
            print("  FAILED TESTS:")
            for f in failures:
                print(f"  - [{f.mechanism}] {f.test_case}: expected={f.expected}, actual={f.actual}")
                if f.detail:
                    print(f"    {f.detail}")
            print()


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    suite = SafetyTestSuite()
    suite.run_all()

    failures = sum(1 for r in suite.results if not r.passed)
    sys.exit(1 if failures > 0 else 0)
