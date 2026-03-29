#!/usr/bin/env python3
"""
Ripple Engine Contradiction Stress Test
========================================

Tests how a simulated ripple engine handles contradictions, corrections,
and knowledge override. The critical risk: new information could incorrectly
override valid old information, silently degrading the brain's knowledge.

This test:
1. Creates test nodes in a COPY of the brain DB
2. Generates enrichments via Gemma 2B (local LLM)
3. Simulates ripple-on-encode: find neighbors, assess impact, apply changes
4. Records whether the LLM assessment was correct
5. Checks if old nodes remain findable after ripple

Requirements:
- Brain DB copy at /tmp/brain_contradiction_test.db
- Arctic embedder model at ~/brain/model-package/brain_embedding/model
- Ollama running with gemma2:2b

Usage:
    python3 tests/test_contradiction_handling.py
"""

import json
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Configuration ──
TEST_DB = "/tmp/brain_contradiction_test.db"
OLLAMA_BIN = "/Applications/Ollama.app/Contents/Resources/ollama"
OLLAMA_MODEL = "gemma2:2b"
EMBEDDER_MODEL_PATH = os.path.expanduser("~/brain/model-package/brain_embedding/model")

# ── Impact assessment prompt for the LLM ──
IMPACT_ASSESSMENT_PROMPT = """You are assessing how a NEW memory relates to an EXISTING memory in a knowledge graph.

EXISTING memory:
Title: "{old_title}"
Content: "{old_content}"
Confidence: {old_confidence}
Locked: {old_locked}

NEW memory:
Title: "{new_title}"
Content: "{new_content}"

Classify the relationship as EXACTLY ONE of:
- VALIDATES: New memory confirms or agrees with the existing memory
- CONTRADICTS: New memory directly conflicts with or disproves the existing memory
- EXTENDS: New memory adds nuance, exceptions, or updates without contradicting
- UNRELATED: Memories are about different topics

Respond with EXACTLY one line in this format:
ASSESSMENT: <VALIDATES|CONTRADICTS|EXTENDS|UNRELATED>
CONFIDENCE_DELTA: <float between -0.5 and +0.2>
REASON: <one sentence>

Rules:
- VALIDATES should increase confidence (+0.05 to +0.15)
- CONTRADICTS should decrease confidence (-0.1 to -0.5)
- EXTENDS should slightly adjust confidence (-0.05 to +0.05)
- UNRELATED should not change confidence (0.0)
- Locked nodes should NOT be marked CONTRADICTS unless the new info is a direct factual correction
- Temporal updates (upgraded to version X) are EXTENDS, not CONTRADICTS
- Exceptions to rules are EXTENDS, not CONTRADICTS"""


# ── Embedder Setup ──
def load_embedder():
    """Load the Arctic embedder directly."""
    from servers import embedder
    config = {
        'model_name': 'Snowflake/snowflake-arctic-embed-m-v1.5',
        'dim': 768,
        'pooling': 'cls',
        'model_file': 'onnx/model.onnx',
        'model_path': EMBEDDER_MODEL_PATH,
    }
    embedder.load_model(config)
    if not embedder.is_ready():
        print("[FATAL] Embedder failed to load. Check model path.")
        sys.exit(1)
    print(f"[OK] Embedder loaded: {embedder.stats['model_name']} (dim={embedder.stats['embedding_dim']})")
    return embedder


# ── Ollama LLM ──
def llm_generate(prompt: str, temperature: float = 0.1) -> str:
    """Call Ollama gemma2:2b for impact assessment."""
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "ERROR: Ollama timed out"
    except Exception as e:
        return f"ERROR: {e}"


def parse_llm_assessment(response: str) -> Dict[str, Any]:
    """Parse the LLM's impact assessment response."""
    result = {
        'assessment': 'UNKNOWN',
        'confidence_delta': 0.0,
        'reason': '',
        'raw_response': response,
        'parse_error': None,
    }

    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.upper().startswith('ASSESSMENT:'):
            val = line.split(':', 1)[1].strip().upper()
            for valid in ('VALIDATES', 'CONTRADICTS', 'EXTENDS', 'UNRELATED'):
                if valid in val:
                    result['assessment'] = valid
                    break
        elif line.upper().startswith('CONFIDENCE_DELTA:'):
            try:
                val = line.split(':', 1)[1].strip()
                # Handle cases where LLM adds extra text after the number
                val = val.split()[0] if val.split() else val
                result['confidence_delta'] = float(val)
            except (ValueError, IndexError) as e:
                result['parse_error'] = f"Failed to parse confidence_delta: {val}"
        elif line.upper().startswith('REASON:'):
            result['reason'] = line.split(':', 1)[1].strip()

    # Clamp confidence delta
    result['confidence_delta'] = max(-0.5, min(0.2, result['confidence_delta']))

    return result


# ── Test Node Management ──
@dataclass
class TestNode:
    """A test node with all metadata."""
    id: str
    type: str
    title: str
    content: str
    confidence: float
    locked: bool = False
    keywords: str = ""
    embedding: Optional[bytes] = None
    enrichments: Dict[str, bytes] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"test_{uuid.uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class RippleResult:
    """Result of a ripple impact assessment."""
    scenario_name: str
    old_node_id: str
    new_node_id: str
    llm_assessment: str
    expected_assessment: str
    correct: bool
    confidence_before: float
    confidence_after: float
    confidence_delta: float
    node_still_findable: bool
    findability_rank: int  # Position in top-10 results (0 = not found)
    llm_reason: str
    llm_raw: str
    edges_created: List[str] = field(default_factory=list)


class ContradictionTestHarness:
    """Test harness for ripple engine contradiction handling."""

    def __init__(self, db_path: str, emb):
        self.db_path = db_path
        self.emb = emb
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.results: List[RippleResult] = []
        self._test_node_ids: List[str] = []

    def cleanup_test_nodes(self):
        """Remove all test nodes we created."""
        for nid in self._test_node_ids:
            self.conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))
            self.conn.execute("DELETE FROM edges WHERE source_id = ? OR target_id = ?", (nid, nid))
            self.conn.execute("DELETE FROM node_embeddings WHERE node_id = ?", (nid,))
            self.conn.execute("DELETE FROM node_enrichments WHERE node_id = ?", (nid,))
            self.conn.execute("DELETE FROM node_vectors WHERE node_id = ?", (nid,))
        self.conn.commit()

    def insert_test_node(self, node: TestNode) -> str:
        """Insert a test node into the DB with embedding."""
        ts = node.created_at or datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            '''INSERT OR REPLACE INTO nodes
               (id, type, title, content, keywords, activation, stability,
                locked, confidence, recency_score, emotion, emotion_label,
                last_accessed, created_at, updated_at, archived)
               VALUES (?, ?, ?, ?, ?, 1.0, 1.0, ?, ?, 1.0, 0, 'neutral', ?, ?, ?, 0)''',
            (node.id, node.type, node.title, node.content, node.keywords,
             1 if node.locked else 0, node.confidence, ts, ts, ts)
        )

        # Embed and store
        embed_text = f"{node.title} {node.content}"
        blob = self.emb.embed(embed_text)
        if blob:
            self.conn.execute(
                'INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
                (node.id, blob, 'snowflake-arctic-embed-m-v1.5', ts)
            )
            node.embedding = blob

        self.conn.commit()
        self._test_node_ids.append(node.id)
        return node.id

    def create_edge(self, source_id: str, target_id: str, relation: str = 'related', weight: float = 0.5):
        """Create an edge between two nodes."""
        ts = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            '''INSERT OR REPLACE INTO edges
               (source_id, target_id, relation, weight, created_at, last_strengthened)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (source_id, target_id, relation, weight, ts, ts)
        )
        self.conn.commit()

    def generate_enrichments(self, node: TestNode) -> Dict[str, str]:
        """Use Gemma 2B to generate Q/A/B/K enrichments for a node."""
        prompt = f"""Generate enrichments for this memory node.

Title: "{node.title}"
Content: "{node.content}"

Generate exactly these lines, no explanations:
Q: [one question a user would naturally ask that leads to this node]
A: [3-5 word anchor phrase]
B: [one sentence connecting this node to related topics]
K: [5 comma-separated keywords]"""

        response = llm_generate(prompt)
        enrichments = {}
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Q:'):
                enrichments['question'] = line[2:].strip()
            elif line.startswith('A:'):
                enrichments['anchor'] = line[2:].strip()
            elif line.startswith('B:'):
                enrichments['bridge'] = line[2:].strip()
            elif line.startswith('K:'):
                enrichments['keywords'] = line[2:].strip()

        # Store enrichments with embeddings
        for vtype, text in enrichments.items():
            if text:
                blob = self.emb.embed(text)
                if blob:
                    eid = uuid.uuid4().hex[:16]
                    ts = datetime.now(timezone.utc).isoformat()
                    self.conn.execute(
                        '''INSERT OR REPLACE INTO node_enrichments
                           (id, node_id, vector_type, text, embedding, model, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)''',
                        (eid, node.id, vtype, text, blob, 'snowflake-arctic-embed-m-v1.5', ts)
                    )
                    node.enrichments[vtype] = blob
        self.conn.commit()
        return enrichments

    def find_neighbors_by_embedding(self, node: TestNode, limit: int = 10) -> List[Tuple[str, float]]:
        """Find similar nodes using cosine similarity on embeddings."""
        if not node.embedding:
            return []

        cursor = self.conn.execute(
            '''SELECT ne.node_id, ne.embedding FROM node_embeddings ne
               JOIN nodes n ON n.id = ne.node_id
               WHERE n.archived = 0 AND ne.node_id != ?''',
            (node.id,)
        )

        scored = []
        for nid, blob in cursor.fetchall():
            if blob:
                sim = self.emb.cosine_similarity(node.embedding, blob)
                scored.append((nid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def assess_impact(self, old_node: TestNode, new_node: TestNode) -> Dict[str, Any]:
        """Use Gemma 2B to assess impact of new node on old node."""
        prompt = IMPACT_ASSESSMENT_PROMPT.format(
            old_title=old_node.title,
            old_content=old_node.content,
            old_confidence=old_node.confidence,
            old_locked="Yes" if old_node.locked else "No",
            new_title=new_node.title,
            new_content=new_node.content,
        )
        response = llm_generate(prompt)
        return parse_llm_assessment(response)

    def check_findability(self, node_id: str, test_queries: List[str], top_k: int = 10) -> Tuple[bool, int]:
        """Check if a node appears in top-k results for any of the test queries.

        Returns (found, best_rank) where best_rank is 1-indexed position (0 = not found).
        Also checks enrichment vectors.
        """
        best_rank = 0
        found = False

        for query in test_queries:
            query_vec = self.emb.embed(query)
            if not query_vec:
                continue

            # Check node embeddings
            cursor = self.conn.execute(
                '''SELECT ne.node_id, ne.embedding FROM node_embeddings ne
                   JOIN nodes n ON n.id = ne.node_id
                   WHERE n.archived = 0'''
            )
            scored = []
            for nid, blob in cursor.fetchall():
                if blob:
                    sim = self.emb.cosine_similarity(query_vec, blob)
                    scored.append((nid, sim, 'node'))

            # Also check enrichment vectors
            cursor = self.conn.execute(
                '''SELECT node_id, embedding FROM node_enrichments
                   WHERE embedding IS NOT NULL'''
            )
            for nid, blob in cursor.fetchall():
                if blob:
                    sim = self.emb.cosine_similarity(query_vec, blob)
                    scored.append((nid, sim, 'enrichment'))

            # Deduplicate by node_id (keep best score)
            best_by_node = {}
            for nid, sim, source in scored:
                if nid not in best_by_node or sim > best_by_node[nid]:
                    best_by_node[nid] = sim

            # Sort and find rank
            ranking = sorted(best_by_node.items(), key=lambda x: x[1], reverse=True)
            for rank_idx, (nid, sim) in enumerate(ranking[:top_k]):
                if nid == node_id:
                    pos = rank_idx + 1
                    if not found or pos < best_rank:
                        best_rank = pos
                    found = True
                    break

        return found, best_rank

    def apply_ripple(self, old_node: TestNode, assessment: Dict[str, Any]) -> float:
        """Apply confidence changes from ripple assessment. Returns new confidence."""
        delta = assessment['confidence_delta']
        new_conf = old_node.confidence + delta

        # Current behavior: no floor, no ceiling protection
        new_conf = max(0.0, min(1.0, new_conf))

        # Apply to DB
        self.conn.execute(
            'UPDATE nodes SET confidence = ?, updated_at = ? WHERE id = ?',
            (new_conf, datetime.now(timezone.utc).isoformat(), old_node.id)
        )
        self.conn.commit()
        return new_conf

    def run_scenario(self, name: str, old_nodes: List[TestNode], new_node: TestNode,
                     expected_assessments: Dict[str, str],
                     test_queries: Dict[str, List[str]],
                     edges: List[Tuple[str, str, str, float]] = None):
        """Run a single contradiction scenario.

        Args:
            name: Scenario name
            old_nodes: Existing nodes to create
            new_node: New node being encoded
            expected_assessments: {old_node_id: expected assessment}
            test_queries: {old_node_id: [queries to check findability]}
            edges: [(source, target, relation, weight)] edges to create
        """
        print(f"\n{'='*70}")
        print(f"SCENARIO: {name}")
        print(f"{'='*70}")

        # 1. Create old nodes
        for node in old_nodes:
            self.insert_test_node(node)
            print(f"  [+] Created old node: {node.id} — '{node.title}' (conf={node.confidence}, locked={node.locked})")

        # 2. Create edges
        if edges:
            for src, tgt, rel, w in edges:
                self.create_edge(src, tgt, rel, w)
                print(f"  [+] Edge: {src} —[{rel} {w}]→ {tgt}")

        # 3. Generate enrichments for old nodes
        for node in old_nodes:
            enrichments = self.generate_enrichments(node)
            print(f"  [+] Enrichments for {node.id}: {list(enrichments.keys())}")

        # 4. Create new node
        self.insert_test_node(new_node)
        print(f"  [+] Created new node: {new_node.id} — '{new_node.title}'")

        # 5. For each old node, assess impact
        for old_node in old_nodes:
            print(f"\n  --- Assessing impact on: '{old_node.title}' ---")

            assessment = self.assess_impact(old_node, new_node)
            print(f"  LLM assessment: {assessment['assessment']}")
            print(f"  LLM delta: {assessment['confidence_delta']}")
            print(f"  LLM reason: {assessment['reason']}")
            if assessment['parse_error']:
                print(f"  PARSE ERROR: {assessment['parse_error']}")

            # 6. Apply confidence change
            conf_before = old_node.confidence
            conf_after = self.apply_ripple(old_node, assessment)
            print(f"  Confidence: {conf_before:.2f} → {conf_after:.2f} (delta={assessment['confidence_delta']:.2f})")

            # 7. Create edge between new and old
            edge_relation = assessment['assessment'].lower()
            self.create_edge(new_node.id, old_node.id, edge_relation, 0.5)

            # 8. Check findability
            queries = test_queries.get(old_node.id, [old_node.title])
            found, rank = self.check_findability(old_node.id, queries)
            print(f"  Findable: {'YES' if found else 'NO'} (rank={rank})")

            # 9. Record result
            expected = expected_assessments.get(old_node.id, 'UNKNOWN')
            correct = assessment['assessment'] == expected

            result = RippleResult(
                scenario_name=name,
                old_node_id=old_node.id,
                new_node_id=new_node.id,
                llm_assessment=assessment['assessment'],
                expected_assessment=expected,
                correct=correct,
                confidence_before=conf_before,
                confidence_after=conf_after,
                confidence_delta=assessment['confidence_delta'],
                node_still_findable=found,
                findability_rank=rank,
                llm_reason=assessment['reason'],
                llm_raw=assessment['raw_response'],
                edges_created=[f"{new_node.id}→{old_node.id} ({edge_relation})"],
            )
            self.results.append(result)

            status = "CORRECT" if correct else f"WRONG (expected {expected})"
            print(f"  Result: {status}")

    def print_summary(self):
        """Print summary table and analysis."""
        print(f"\n\n{'='*100}")
        print("RESULTS SUMMARY")
        print(f"{'='*100}")
        print(f"{'Scenario':<35} {'LLM':<12} {'Expected':<12} {'OK?':<6} {'Conf':<12} {'Find?':<6} {'Rank':<5}")
        print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*6} {'-'*12} {'-'*6} {'-'*5}")

        for r in self.results:
            conf_str = f"{r.confidence_before:.2f}→{r.confidence_after:.2f}"
            ok_str = "YES" if r.correct else "NO"
            find_str = "YES" if r.node_still_findable else "NO"
            print(f"{r.scenario_name:<35} {r.llm_assessment:<12} {r.expected_assessment:<12} "
                  f"{ok_str:<6} {conf_str:<12} {find_str:<6} {r.findability_rank:<5}")

        # Statistics
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        findable = sum(1 for r in self.results if r.node_still_findable)
        wrong_assessments = [r for r in self.results if not r.correct]

        print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")
        print(f"Findability: {findable}/{total} ({100*findable/total:.0f}%)")

        if wrong_assessments:
            print(f"\n{'='*70}")
            print("FAILURE MODES")
            print(f"{'='*70}")
            for r in wrong_assessments:
                print(f"\n  Scenario: {r.scenario_name}")
                print(f"  Expected: {r.expected_assessment}, Got: {r.llm_assessment}")
                print(f"  LLM Reason: {r.llm_reason}")
                print(f"  --- LLM Raw Response ---")
                for line in r.llm_raw.split('\n')[:8]:
                    print(f"    {line}")

        # Findability failures
        unfindable = [r for r in self.results if not r.node_still_findable]
        if unfindable:
            print(f"\n{'='*70}")
            print("FINDABILITY FAILURES (nodes lost after ripple)")
            print(f"{'='*70}")
            for r in unfindable:
                print(f"  {r.scenario_name}: {r.old_node_id} — conf {r.confidence_before:.2f}→{r.confidence_after:.2f}")

        # Prompts that caused wrong assessments
        if wrong_assessments:
            print(f"\n{'='*70}")
            print("PROMPTS THAT CAUSED WRONG ASSESSMENTS")
            print(f"{'='*70}")
            for r in wrong_assessments:
                # Reconstruct the prompt from the result
                old_row = self.conn.execute(
                    'SELECT title, content, confidence, locked FROM nodes WHERE id = ?',
                    (r.old_node_id,)
                ).fetchone()
                new_row = self.conn.execute(
                    'SELECT title, content FROM nodes WHERE id = ?',
                    (r.new_node_id,)
                ).fetchone()
                if old_row and new_row:
                    print(f"\n  --- {r.scenario_name} ---")
                    print(f"  Old: '{old_row[0]}' | New: '{new_row[0]}'")
                    print(f"  LLM said: {r.llm_assessment} (expected {r.expected_assessment})")
                    print(f"  Key issue: LLM {'over-classified' if r.llm_assessment == 'CONTRADICTS' else 'under-classified'} the relationship")

        return {
            'total': total,
            'correct': correct,
            'findable': findable,
            'wrong_assessments': [(r.scenario_name, r.llm_assessment, r.expected_assessment) for r in wrong_assessments],
            'unfindable': [(r.scenario_name, r.old_node_id) for r in unfindable],
        }


# ── Scenario Definitions ──

def run_all_scenarios(harness: ContradictionTestHarness):
    """Run all 8 contradiction scenarios."""

    # ── Scenario 1: Legitimate Correction (actually agreement) ──
    old1 = TestNode(
        id="test_s1_old",
        type="rule",
        title="React hooks can't be used in class components",
        content="React hooks (useState, useEffect, etc.) are restricted to functional components only. They cannot be called inside class components.",
        confidence=0.85,
        locked=True,
        keywords="react hooks class components functional",
    )
    new1 = TestNode(
        id="test_s1_new",
        type="rule",
        title="React hooks work ONLY in functional components, not class components",
        content="React hooks are designed exclusively for functional components. They do not work in class components by design.",
        confidence=0.90,
        keywords="react hooks functional components class",
    )
    harness.run_scenario(
        name="S1: Legitimate correction",
        old_nodes=[old1],
        new_node=new1,
        expected_assessments={"test_s1_old": "VALIDATES"},
        test_queries={"test_s1_old": [
            "can I use hooks in class components",
            "react hooks class vs functional",
            "where do hooks work in react",
        ]},
    )

    # ── Scenario 2: Genuine Contradiction (partial) ──
    old2 = TestNode(
        id="test_s2_old",
        type="convention",
        title="We use REST API for all endpoints",
        content="All API endpoints in the application use REST architecture. Every endpoint follows REST conventions for CRUD operations.",
        confidence=0.80,
        locked=False,
        keywords="REST API endpoints architecture CRUD",
    )
    new2 = TestNode(
        id="test_s2_new",
        type="decision",
        title="Switched to GraphQL for the dashboard endpoints",
        content="Dashboard endpoints have been migrated from REST to GraphQL to reduce over-fetching. Other endpoints remain REST.",
        confidence=0.90,
        keywords="GraphQL dashboard endpoints migration REST",
    )
    harness.run_scenario(
        name="S2: Genuine contradiction",
        old_nodes=[old2],
        new_node=new2,
        expected_assessments={"test_s2_old": "EXTENDS"},
        test_queries={"test_s2_old": [
            "what API architecture do we use",
            "REST API endpoints",
            "do we use REST or GraphQL",
        ]},
    )

    # ── Scenario 3: Temporal Update ──
    old3 = TestNode(
        id="test_s3_old",
        type="context",
        title="Current stack: React 17 + Node 16",
        content="The project uses React 17 and Node.js 16 as the current technology stack.",
        confidence=0.70,
        locked=False,
        keywords="react node stack technology version",
        created_at=(datetime.now(timezone.utc) - timedelta(days=180)).isoformat(),
    )
    new3 = TestNode(
        id="test_s3_new",
        type="context",
        title="Upgraded to React 18 + Node 20",
        content="Project has been upgraded from React 17 to React 18 and from Node 16 to Node 20.",
        confidence=0.90,
        keywords="react node stack upgrade version",
    )
    harness.run_scenario(
        name="S3: Temporal update",
        old_nodes=[old3],
        new_node=new3,
        expected_assessments={"test_s3_old": "EXTENDS"},
        test_queries={"test_s3_old": [
            "what was the old tech stack",
            "react 17 node 16",
            "previous stack before upgrade",
        ]},
    )

    # ── Scenario 4: Nuanced Disagreement ──
    old4 = TestNode(
        id="test_s4_old",
        type="rule",
        title="Tom prefers functional components over class components",
        content="Tom strongly prefers using functional components with hooks instead of class components in React codebases.",
        confidence=0.90,
        locked=True,
        keywords="Tom preference functional class components react",
    )
    new4 = TestNode(
        id="test_s4_new",
        type="decision",
        title="Used a class component for ErrorBoundary because React requires it",
        content="ErrorBoundary must be a class component because React doesn't have a hook equivalent for componentDidCatch. This is an exception to the functional component preference.",
        confidence=0.85,
        keywords="ErrorBoundary class component react exception",
    )
    harness.run_scenario(
        name="S4: Nuanced disagreement",
        old_nodes=[old4],
        new_node=new4,
        expected_assessments={"test_s4_old": "EXTENDS"},
        test_queries={"test_s4_old": [
            "does Tom prefer functional or class components",
            "Tom component preference react",
            "functional vs class components preference",
        ]},
    )

    # ── Scenario 5: Cascade Risk ──
    old5a = TestNode(
        id="test_s5_a",
        type="context",
        title="Architecture is monolith",
        content="The system architecture is a monolithic application. All services run in a single process.",
        confidence=0.85,
        locked=False,
        keywords="architecture monolith single process",
    )
    old5b = TestNode(
        id="test_s5_b",
        type="lesson",
        title="Monolith was fine for v1",
        content="The monolithic architecture worked well for version 1 of the product. It was simple and fast to develop.",
        confidence=0.70,
        locked=False,
        keywords="monolith v1 architecture lesson",
    )
    old5c = TestNode(
        id="test_s5_c",
        type="mechanism",
        title="Database is shared between all services",
        content="All services share a single PostgreSQL database. There is no service-level data isolation.",
        confidence=0.75,
        locked=False,
        keywords="database shared services postgresql",
    )
    new5 = TestNode(
        id="test_s5_new",
        type="decision",
        title="Separated into microservices",
        content="The monolith has been decomposed into microservices. Each service now runs independently.",
        confidence=0.90,
        keywords="microservices architecture decomposition",
    )
    harness.run_scenario(
        name="S5: Cascade risk",
        old_nodes=[old5a, old5b, old5c],
        new_node=new5,
        expected_assessments={
            "test_s5_a": "CONTRADICTS",
            "test_s5_b": "VALIDATES",  # It WAS fine — that's still true
            "test_s5_c": "EXTENDS",    # May or may not have changed
        },
        test_queries={
            "test_s5_a": ["was it a monolith", "original architecture", "monolith architecture"],
            "test_s5_b": ["was monolith good for v1", "monolith worked", "v1 architecture"],
            "test_s5_c": ["shared database", "database architecture", "service data isolation"],
        },
        edges=[
            ("test_s5_a", "test_s5_b", "supports", 0.6),
            ("test_s5_a", "test_s5_c", "implies", 0.7),
            ("test_s5_b", "test_s5_c", "related", 0.4),
        ],
    )

    # ── Scenario 6: Operator Override Protection ──
    old6 = TestNode(
        id="test_s6_old",
        type="rule",
        title="NEVER modify sacred systems without benchmarks",
        content="Before changing any sacred system (embedder, recall, encoding, precision, hook output), build test harness FIRST. Benchmark with real-world cases. Only ship after benchmarks prove no regression.",
        confidence=0.95,
        locked=True,
        keywords="sacred systems benchmarks testing rule protection",
    )
    new6 = TestNode(
        id="test_s6_new",
        type="lesson",
        title="Quick-fixed the embedder without benchmarks because production was down",
        content="Production was broken due to embedder failure. Applied emergency fix without running full benchmark suite. Exception to the benchmark-first rule due to urgency.",
        confidence=0.80,
        keywords="embedder fix production emergency benchmarks exception",
    )
    harness.run_scenario(
        name="S6: Operator override protection",
        old_nodes=[old6],
        new_node=new6,
        expected_assessments={"test_s6_old": "EXTENDS"},
        test_queries={"test_s6_old": [
            "do I need benchmarks before changing sacred systems",
            "benchmark first rule",
            "sacred system modification policy",
        ]},
    )

    # ── Scenario 7: Confidence Floor ──
    old7 = TestNode(
        id="test_s7_old",
        type="context",
        title="Python 3.9 is the minimum version",
        content="The project requires Python 3.9 as the minimum supported version.",
        confidence=0.30,
        locked=False,
        keywords="python version minimum 3.9 requirement",
    )
    new7 = TestNode(
        id="test_s7_new",
        type="decision",
        title="Added type hints requiring Python 3.10+",
        content="New type hint syntax (X | Y instead of Union[X, Y]) requires Python 3.10 or higher. The minimum version has been bumped.",
        confidence=0.85,
        keywords="python version 3.10 type hints upgrade",
    )
    harness.run_scenario(
        name="S7: Confidence floor",
        old_nodes=[old7],
        new_node=new7,
        expected_assessments={"test_s7_old": "CONTRADICTS"},
        test_queries={"test_s7_old": [
            "what was the minimum python version",
            "python 3.9 requirement",
            "old python version requirement",
        ]},
    )

    # ── Scenario 8: Multiple Rapid-Fire Updates ──
    # We simulate this as sequential encoding
    node8_base = TestNode(
        id="test_s8_base",
        type="context",
        title="API latency is 200ms",
        content="The API response latency is approximately 200 milliseconds under normal load.",
        confidence=0.80,
        locked=False,
        keywords="API latency 200ms performance",
    )
    harness.insert_test_node(node8_base)
    harness.generate_enrichments(node8_base)
    print(f"\n{'='*70}")
    print("SCENARIO: S8: Multiple rapid-fire updates")
    print(f"{'='*70}")
    print(f"  [+] Created base node: {node8_base.id} — '{node8_base.title}' (conf={node8_base.confidence})")

    # Update 1: Improvement
    node8_update1 = TestNode(
        id="test_s8_update1",
        type="context",
        title="API latency improved to 50ms",
        content="API latency has been optimized from 200ms to approximately 50ms through caching and query optimization.",
        confidence=0.85,
        keywords="API latency 50ms performance improvement",
    )
    harness.insert_test_node(node8_update1)

    assessment1 = harness.assess_impact(node8_base, node8_update1)
    conf_after_1 = harness.apply_ripple(node8_base, assessment1)
    print(f"  Update 1: {assessment1['assessment']} (delta={assessment1['confidence_delta']:.2f})")
    print(f"  Base conf: {node8_base.confidence:.2f} → {conf_after_1:.2f}")

    # Update 2: Regression
    node8_update2 = TestNode(
        id="test_s8_update2",
        type="context",
        title="API latency regressed to 500ms",
        content="API latency regressed to 500ms after deploying the new authentication middleware. Investigating.",
        confidence=0.85,
        keywords="API latency 500ms regression performance",
    )
    harness.insert_test_node(node8_update2)

    # Assess against BOTH previous nodes
    assessment2a = harness.assess_impact(node8_base, node8_update2)
    assessment2b = harness.assess_impact(node8_update1, node8_update2)

    # Apply to base
    node8_base.confidence = conf_after_1  # Track current state
    conf_after_2 = harness.apply_ripple(node8_base, assessment2a)
    print(f"  Update 2 (vs base): {assessment2a['assessment']} (delta={assessment2a['confidence_delta']:.2f})")
    print(f"  Base conf: {conf_after_1:.2f} → {conf_after_2:.2f}")

    # Apply to update1
    node8_update1_conf = node8_update1.confidence
    conf_update1_after = harness.apply_ripple(node8_update1, assessment2b)
    print(f"  Update 2 (vs update1): {assessment2b['assessment']} (delta={assessment2b['confidence_delta']:.2f})")
    print(f"  Update1 conf: {node8_update1_conf:.2f} → {conf_update1_after:.2f}")

    # Check findability of all three
    for node, label, queries in [
        (node8_base, "base (200ms)", ["original API latency", "API was 200ms"]),
        (node8_update1, "update1 (50ms)", ["API latency improvement", "50ms latency"]),
        (node8_update2, "update2 (500ms)", ["API latency regression", "current API latency"]),
    ]:
        found, rank = harness.check_findability(node.id, queries)
        print(f"  Findable {label}: {'YES' if found else 'NO'} (rank={rank})")

    # Record results for scenario 8
    result_s8 = RippleResult(
        scenario_name="S8: Rapid-fire (base)",
        old_node_id=node8_base.id,
        new_node_id=node8_update2.id,
        llm_assessment=assessment2a['assessment'],
        expected_assessment="EXTENDS",
        correct=assessment2a['assessment'] in ('EXTENDS', 'CONTRADICTS'),  # Both acceptable
        confidence_before=0.80,
        confidence_after=conf_after_2,
        confidence_delta=conf_after_2 - 0.80,
        node_still_findable=harness.check_findability(node8_base.id, ["original API latency 200ms"])[0],
        findability_rank=harness.check_findability(node8_base.id, ["original API latency 200ms"])[1],
        llm_reason=assessment2a['reason'],
        llm_raw=assessment2a['raw_response'],
    )
    harness.results.append(result_s8)

    # Check for double-ripple issue
    print(f"\n  --- Double-ripple check ---")
    final_base_conf = harness.conn.execute(
        'SELECT confidence FROM nodes WHERE id = ?', (node8_base.id,)
    ).fetchone()[0]
    print(f"  Final base confidence: {final_base_conf:.2f}")
    if final_base_conf <= 0.0:
        print(f"  WARNING: Base node confidence hit zero — double-ripple destroyed it!")
    elif final_base_conf < 0.10:
        print(f"  WARNING: Base node confidence near zero ({final_base_conf:.2f}) — dangerously low")


def propose_safeguards(results: List[RippleResult]) -> List[Dict[str, Any]]:
    """Analyze results and propose safety mechanisms."""
    safeguards = []

    # 1. Confidence floor
    low_conf = [r for r in results if r.confidence_after < 0.10]
    if low_conf:
        safeguards.append({
            'name': 'Confidence Floor',
            'priority': 'HIGH',
            'description': 'Never allow confidence to drop below 0.05. Nodes at 0.0 become permanently invisible.',
            'implementation': 'In apply_ripple: new_conf = max(0.05, new_conf)',
            'triggered_by': [r.scenario_name for r in low_conf],
        })
    else:
        safeguards.append({
            'name': 'Confidence Floor',
            'priority': 'MEDIUM',
            'description': 'Preventive: ensure confidence never goes below 0.05.',
            'implementation': 'In apply_ripple: new_conf = max(0.05, new_conf)',
            'triggered_by': [],
        })

    # 2. Locked node protection
    wrong_locked = [r for r in results if not r.correct and
                    r.llm_assessment == 'CONTRADICTS' and
                    r.expected_assessment != 'CONTRADICTS']
    safeguards.append({
        'name': 'Locked Node Protection',
        'priority': 'HIGH',
        'description': 'Locked nodes cannot have their confidence reduced by ripple. Mark as EXTENDS instead, log the conflict.',
        'implementation': 'If node.locked and assessment == CONTRADICTS: downgrade to EXTENDS, set delta = 0, log conflict',
        'triggered_by': [r.scenario_name for r in wrong_locked],
    })

    # 3. Operator confirmation threshold
    big_drops = [r for r in results if r.confidence_delta < -0.3]
    safeguards.append({
        'name': 'Operator Confirmation Threshold',
        'priority': 'HIGH',
        'description': 'If confidence would drop > 0.3, require operator confirmation before applying.',
        'implementation': 'If abs(delta) > 0.3: stage the ripple, surface in consciousness signals, wait for approval',
        'triggered_by': [r.scenario_name for r in big_drops],
    })

    # 4. Rate limiting
    safeguards.append({
        'name': 'Ripple Rate Limit',
        'priority': 'MEDIUM',
        'description': 'Max 5 ripple assessments per remember() call. Prevents cascade storms.',
        'implementation': 'Counter in remember(): if ripple_count > 5, queue remaining for idle-time processing',
        'triggered_by': ['S5: Cascade risk', 'S8: Rapid-fire'],
    })

    # 5. Cascade depth limit
    safeguards.append({
        'name': 'Cascade Depth Limit',
        'priority': 'HIGH',
        'description': 'Ripple does NOT trigger re-ripple. Only direct neighbors are assessed.',
        'implementation': 'Set depth=1 flag; ripple callbacks check depth and stop if > 1',
        'triggered_by': ['S5: Cascade risk'],
    })

    # 6. Undo log
    safeguards.append({
        'name': 'Undo Log (Pre-Ripple Snapshot)',
        'priority': 'MEDIUM',
        'description': 'Before any ripple, snapshot {node_id, old_confidence, old_edges} to ripple_log table. Enables rollback.',
        'implementation': 'New table: ripple_log (id, node_id, old_confidence, new_confidence, assessment, session_id, created_at)',
        'triggered_by': ['All scenarios'],
    })

    # 7. Assessment validation
    wrong = [r for r in results if not r.correct]
    if wrong:
        safeguards.append({
            'name': 'Dual-Assessment Validation',
            'priority': 'HIGH',
            'description': f'LLM got {len(wrong)}/{len(results)} wrong. Run assessment twice with temperature variation; only apply if both agree.',
            'implementation': 'Call LLM twice (temp 0.1 and 0.3). If assessments differ, default to EXTENDS with delta=0.',
            'triggered_by': [r.scenario_name for r in wrong],
        })

    # 8. Temporal awareness
    safeguards.append({
        'name': 'Temporal Marker',
        'priority': 'LOW',
        'description': 'When a node is CONTRADICTED by a temporal update, add "superseded_by" edge instead of reducing confidence to zero.',
        'implementation': 'If assessment indicates temporal change, create superseded_by edge, set old confidence to max(0.3, current - 0.2)',
        'triggered_by': ['S3: Temporal update'],
    })

    return safeguards


def main():
    """Main entry point."""
    print("="*70)
    print("RIPPLE ENGINE CONTRADICTION STRESS TEST")
    print("="*70)
    print(f"DB: {TEST_DB}")
    print(f"LLM: {OLLAMA_MODEL} via {OLLAMA_BIN}")
    print(f"Embedder: Arctic v1.5 from {EMBEDDER_MODEL_PATH}")
    print()

    # Verify DB exists
    if not os.path.exists(TEST_DB):
        src = os.path.expanduser("~/AgentsContext/brain/brain.db")
        if os.path.exists(src):
            shutil.copy2(src, TEST_DB)
            print(f"[OK] Copied {src} → {TEST_DB}")
        else:
            print(f"[FATAL] No brain DB at {src}")
            sys.exit(1)

    # Verify Ollama
    try:
        result = subprocess.run([OLLAMA_BIN, "list"], capture_output=True, text=True, timeout=5)
        if OLLAMA_MODEL not in result.stdout:
            print(f"[FATAL] {OLLAMA_MODEL} not available in Ollama")
            sys.exit(1)
        print(f"[OK] Ollama: {OLLAMA_MODEL} available")
    except Exception as e:
        print(f"[FATAL] Ollama not available: {e}")
        sys.exit(1)

    # Load embedder
    emb = load_embedder()

    # Create harness
    harness = ContradictionTestHarness(TEST_DB, emb)

    try:
        # Run all scenarios
        run_all_scenarios(harness)

        # Print summary
        summary = harness.print_summary()

        # Propose safeguards
        safeguards = propose_safeguards(harness.results)
        print(f"\n\n{'='*70}")
        print("PROPOSED SAFETY MECHANISMS")
        print(f"{'='*70}")
        for sg in sorted(safeguards, key=lambda x: {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}[x['priority']]):
            print(f"\n  [{sg['priority']}] {sg['name']}")
            print(f"  {sg['description']}")
            print(f"  Implementation: {sg['implementation']}")
            if sg['triggered_by']:
                print(f"  Triggered by: {', '.join(sg['triggered_by'])}")

        # Save results as JSON
        results_path = "/tmp/contradiction_test_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'summary': summary,
                'results': [
                    {
                        'scenario': r.scenario_name,
                        'old_node_id': r.old_node_id,
                        'new_node_id': r.new_node_id,
                        'llm_assessment': r.llm_assessment,
                        'expected_assessment': r.expected_assessment,
                        'correct': r.correct,
                        'confidence_before': r.confidence_before,
                        'confidence_after': r.confidence_after,
                        'confidence_delta': r.confidence_delta,
                        'node_still_findable': r.node_still_findable,
                        'findability_rank': r.findability_rank,
                        'llm_reason': r.llm_reason,
                        'llm_raw': r.llm_raw[:500],
                    }
                    for r in harness.results
                ],
                'safeguards': safeguards,
            }, f, indent=2)
        print(f"\n[OK] Results saved to {results_path}")

    finally:
        # Cleanup test nodes from the copy
        harness.cleanup_test_nodes()
        harness.conn.close()
        print(f"\n[OK] Test nodes cleaned up from {TEST_DB}")


if __name__ == '__main__':
    main()
