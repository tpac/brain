"""tmemory Brain Engine v7 — Python Port

Hebbian learning, Ebbinghaus decay, synaptic pruning, spreading activation.

PART 1: Constants, constructor, and core helper methods.
Part 2 (remember, recall, dream, consolidate, suggest, context_boot, health_check, etc.)
continues below in separate files.

Architecture: sqlite3 (WAL mode, FK on), fire-and-forget async embeddings,
TF-IDF semantic scoring, intent detection, temporal awareness.
"""

import sqlite3
import math
import re
import uuid
import json
import time
import os
import struct
import threading
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Set
from .schema import ensure_schema, BRAIN_VERSION, BRAIN_VERSION_KEY, NODE_TYPES
from . import embedder


# ═══════════════════════════════════════════════════════════════
# CONSTANTS: Decay rates by node type (hours until weight halves)
# ═══════════════════════════════════════════════════════════════

DECAY_HALF_LIFE = {
    'person': 720,      # 30 days
    'project': 720,     # 30 days
    'object': 720,      # 30 days — structured entity groupings persist like projects
    'decision': float('inf'),  # never (unless not locked)
    'rule': float('inf'),      # never
    'concept': 168,     # 7 days
    'task': 48,         # 2 days
    'file': 168,        # 7 days
    'context': 24,      # 1 day
    'intuition': 12,    # 12 hours — fleeting unless reinforced
    'procedure': float('inf'),  # never — user/system routines persist
    'thought': 3,       # 3 hours — brain's own observations, prune very fast, survive only if touched
}

# Stability floor: nodes accessed >= this many times get a minimum retention
# Prevents well-reinforced knowledge from fully decaying during long absences
STABILITY_FLOOR_ACCESS_THRESHOLD = 5  # need 5+ accesses to earn a floor
STABILITY_FLOOR_RETENTION = 0.3       # minimum 30% retention for qualifying nodes

# Hebbian learning rate
LEARNING_RATE = 0.2
MAX_WEIGHT = 1.0
PRUNE_THRESHOLD = 0.05
SPREAD_DECAY = 0.5
MAX_HOPS = 3
MAX_NEIGHBORS = 50
STABILITY_BOOST = 1.5

# ─── v2: Recency scoring weights ───
# v3: Rebalanced to include emotion (total = 1.0)
RECENCY_WEIGHT = 0.30     # 30% from recency
RELEVANCE_WEIGHT = 0.35   # 35% from keyword/graph relevance
FREQUENCY_WEIGHT = 0.10   # 10% from access frequency
EMOTION_WEIGHT = 0.25     # 25% from emotional intensity

# ─── v3: Emotion constants ───
# Emotion scale: 0.0 = neutral, 1.0 = maximum emotional intensity
# Sign is stored separately in emotion_label (positive/negative context)
EMOTION_FLOOR = 0.3       # Neutral nodes still get a base emotion score
EMOTION_DECAY_RATE = 0.95 # Emotion fades slowly (×0.95 per day)

# ─── v3: Dreaming constants ───
DREAM_WALK_LENGTH = 5     # Random walk steps per dream
DREAM_COUNT = 3           # Number of dreams per session
DREAM_MIN_NOVELTY = 2     # Min hops between dream seeds (ensures novelty)

# Recency time bands (hours → score multiplier)
RECENCY_BANDS = [
    {'maxHours': 1,     'score': 1.0},    # last hour = full recency
    {'maxHours': 6,     'score': 0.9},    # last 6h
    {'maxHours': 24,    'score': 0.75},   # last day
    {'maxHours': 72,    'score': 0.5},    # last 3 days
    {'maxHours': 168,   'score': 0.3},    # last week
    {'maxHours': 720,   'score': 0.15},   # last month
    {'maxHours': float('inf'), 'score': 0.05}  # older
]

# ─── v2: Progressive access page sizes ───
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
CONTEXT_BOOT_LOCKED_LIMIT = 50  # cap locked nodes in boot to prevent flood
CONTEXT_BOOT_RECALL_LIMIT = 15
CONTEXT_BOOT_RECENT_LIMIT = 10

# ─── v5: TF-IDF constants ───
TFIDF_SEMANTIC_WEIGHT = 0.35  # Weight for semantic (TF-IDF) score in combined recall
TFIDF_KEYWORD_WEIGHT = 0.65   # Weight for existing keyword/spread activation score
TFIDF_STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of',
    'that', 'this', 'it', 'be', 'as', 'by', 'from', 'has', 'was', 'are', 'were', 'been', 'have', 'had',
    'not', 'they', 'their', 'will', 'would', 'can', 'could', 'do', 'does', 'did', 'should', 'may',
    'might', 'shall', 'than', 'into', 'about', 'also', 'its', 'just', 'more', 'other', 'some', 'such',
    'then', 'there', 'these', 'what', 'when', 'where', 'who', 'how', 'all', 'each', 'both', 'few',
    'most', 'any', 'own', 'same', 'over', 'only', 'very', 'after', 'before', 'between', 'under',
    'above', 'out', 'up', 'down', 'use', 'used', 'using', 'new', 'like', 'get', 'set', 'one', 'two'
}

# ─── v5: Intent detection patterns ───
INTENT_PATTERNS = {
    'decision_lookup': re.compile(r'\b(what did (?:we|tom|i) (?:decide|choose|pick)|decision about|decided on)\b', re.IGNORECASE),
    'reasoning_chain': re.compile(r'\b(why did (?:we|i)|reason for|reasoning behind|what led to|how come)\b', re.IGNORECASE),
    'state_query': re.compile(r"\b(what(?:'s| is) the (?:current|latest)|status of|state of|where (?:are|is) (?:we|it))\b", re.IGNORECASE),
    'temporal': re.compile(r'\b(when did|last (?:week|month|time|session)|this (?:week|month)|before (?:the|we)|after (?:the|we)|yesterday|today|recently|history of|timeline)\b', re.IGNORECASE),
    'correction_lookup': re.compile(r'\b(what mistake|lesson(?:s)? learned|correction|what went wrong|what did (?:we|i) learn|mistakes?\b.*learn|learn(?:ed)? from)\b', re.IGNORECASE),
    'how_to': re.compile(r"\b(how (?:do|does|to|should)|what(?:'s| is) the (?:best|right) way)\b", re.IGNORECASE),
    'list_query': re.compile(r'\b(list (?:all|every)|show me (?:all|every)|what are (?:all|the))\b', re.IGNORECASE),
}

# v5: Intent → type boosts (which node types score higher for each intent)
INTENT_TYPE_BOOSTS = {
    'decision_lookup':   {'decision': 1.5, 'rule': 1.0},
    'reasoning_chain':   {'decision': 1.3, 'rule': 1.2, 'context': 1.1},
    'state_query':       {'context': 1.5, 'project': 1.3, 'task': 1.3, 'object': 1.4},
    'temporal':          {'decision': 1.0, 'context': 1.2},
    'correction_lookup': {'decision': 1.5, 'rule': 1.2},
    'how_to':            {'rule': 1.5, 'decision': 1.2},
    'list_query':        {'rule': 1.0, 'decision': 1.0, 'object': 1.3},
    'general':           {},  # no boost
}

# ─── v5: Temporal parsing patterns ───
TEMPORAL_PATTERNS = [
    {
        'pattern': re.compile(r'\btoday\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\byesterday\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).replace(day=datetime.utcnow().day - 1).isoformat() + 'Z'),
            'before': (datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'),
        }
    },
    {
        'pattern': re.compile(r'\bthis week\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(day=datetime.utcnow().day - datetime.utcnow().weekday(), hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\blast week\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(day=datetime.utcnow().day - datetime.utcnow().weekday() - 7, hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'before': datetime.utcnow().replace(day=datetime.utcnow().day - datetime.utcnow().weekday(), hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
        }
    },
    {
        'pattern': re.compile(r'\bthis month\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\blast month\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(month=datetime.utcnow().month - 1, day=1, hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'before': datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
        }
    },
    {
        'pattern': re.compile(r'\blast (\d+) days?\b', re.IGNORECASE),
        'range_fn': lambda m: {
            'after': datetime.utcnow().replace(day=datetime.utcnow().day - int(m.group(1))).isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\brecently\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(day=datetime.utcnow().day - 3).isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\blast session\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': datetime.utcnow().replace(hour=datetime.utcnow().hour - 6).isoformat() + 'Z'
        }
    },
]

# ─── v6: Edge type definitions ───
# Each edge type has: defaultWeight, whether it decays, and half-life in hours
EDGE_TYPES = {
    'reasoning_step': {'defaultWeight': 0.9, 'decays': False, 'description': 'Step N → Step N+1 in a reasoning chain'},
    'produced': {'defaultWeight': 0.85, 'decays': False, 'description': 'Reasoning chain → Decision it produced'},
    'corrected_by': {'defaultWeight': 0.85, 'decays': False, 'description': 'Correction event → person who corrected'},
    'exemplifies': {'defaultWeight': 0.8, 'decays': True, 'halfLife': 720, 'description': 'Decision → Rule it demonstrates'},
    'part_of': {'defaultWeight': 0.7, 'decays': False, 'description': 'Node → Project or system it belongs to'},
    'depends_on': {'defaultWeight': 0.7, 'decays': False, 'description': 'Node requires another node'},
    'related': {'defaultWeight': 0.5, 'decays': True, 'halfLife': 336, 'description': 'Manual or inferred connection'},
    'co_accessed': {'defaultWeight': 0.3, 'decays': True, 'halfLife': 168, 'description': 'Hebbian — co-recalled in same session'},
    'emergent_bridge': {'defaultWeight': 0.15, 'decays': True, 'halfLife': 72, 'description': 'Auto-discovered from shared neighbors — lightweight, must prove value'},
}

# v6: Reasoning step types
REASONING_STEP_TYPES = ['observation', 'hypothesis', 'attempt', 'evidence', 'failure', 'feedback', 'decision', 'lesson']

# v6: Curiosity thresholds
CURIOSITY_MAX_PROMPTS = 3           # Max curiosity prompts per session
CURIOSITY_CHAIN_GAP_THRESHOLD = 0   # Decisions with 0 reasoning chains are gaps
CURIOSITY_DECAY_WARNING_HOURS = 18  # Flag context nodes within 18h of their 24h half-life


# ═══════════════════════════════════════════════════════════════
# BRAIN CLASS
# ═══════════════════════════════════════════════════════════════

class Brain:
    """
    Core brain engine for tmemory.

    Manages:
    - Node storage (memories, thoughts, rules, etc.)
    - Edge connections (Hebbian learning)
    - Semantic recall (TF-IDF + embeddings)
    - Intent detection and temporal filtering
    - Session activity tracking

    Singleton pattern: Use Brain.get_instance(db_path) to reuse an existing
    warm Brain for the same db_path. Direct __init__ always creates a new instance
    (useful for tests, simulations, and fresh brains).
    """

    # ─── Singleton registry ───
    _instances: Dict[str, 'Brain'] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, db_path: str) -> 'Brain':
        """
        Get or create a singleton Brain for the given db_path.

        Returns an existing warm instance if one is already open for this path,
        avoiding repeated schema checks, TF-IDF rebuilds, and embedder loads.
        Thread-safe.

        Args:
            db_path: Path to brain.db file

        Returns:
            Brain instance (cached or newly created)
        """
        canonical = os.path.realpath(db_path)
        with cls._lock:
            instance = cls._instances.get(canonical)
            if instance is not None:
                # Verify the connection is still alive
                try:
                    instance.conn.execute('SELECT 1')
                    return instance
                except Exception:
                    # Connection died — remove stale entry and recreate
                    del cls._instances[canonical]

            instance = cls(db_path)
            cls._instances[canonical] = instance
            return instance

    @classmethod
    def clear_instances(cls):
        """
        Close and remove all cached Brain instances.
        Useful for test teardown or when switching brain files.
        """
        with cls._lock:
            for path, instance in list(cls._instances.items()):
                try:
                    instance.conn.commit()
                    instance.conn.close()
                except Exception:
                    pass
            cls._instances.clear()

    def __init__(self, db_path: str):
        """
        Initialize Brain with SQLite3 database.

        NOTE: For production use, prefer Brain.get_instance(db_path) which
        reuses warm instances. Direct __init__ always creates a fresh connection
        (appropriate for tests, simulations, and temporary brains).

        Args:
            db_path: Path to brain.db file
        """
        self.db_path = db_path

        # Open SQLite connection with WAL mode for concurrency
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA foreign_keys=ON')

        # Create schema if needed
        ensure_schema(self.conn)

        # Post-schema initialization (TF-IDF rebuild if needed)
        self._post_schema_init()

        # Try to load embedder (optional; graceful degradation if not available)
        try:
            embedder.load_model()
        except Exception as e:
            print(f'[brain] Embedder load failed (optional): {e}')

    def _post_schema_init(self):
        """
        Build TF-IDF index if node_vectors is empty but nodes exist.
        Called after ensure_schema() to handle runtime initialization.
        """
        try:
            cursor = self.conn.execute('SELECT COUNT(*) FROM node_vectors')
            vector_count = cursor.fetchone()[0]

            cursor = self.conn.execute('SELECT COUNT(*) FROM nodes')
            node_count = cursor.fetchone()[0]

            if node_count > 0 and vector_count == 0:
                print('[brain] Building TF-IDF index for existing nodes...')
                self._rebuild_tfidf_index()
                print('[brain] TF-IDF index built.')
        except Exception:
            # Tables might not exist yet on very first run
            pass

    def now(self) -> str:
        """Return current UTC ISO timestamp."""
        return datetime.utcnow().isoformat() + 'Z'

    def _generate_id(self, node_type: str = None) -> str:
        """Generate unique node ID using uuid4 hex."""
        return uuid.uuid4().hex

    # ─── Helper: Recency Scoring ───

    def _recency_score(self, last_accessed: Optional[str]) -> float:
        """
        Compute recency score for a node based on last access time.
        Maps hours ago to RECENCY_BANDS score.

        Args:
            last_accessed: ISO timestamp of last access (or None)

        Returns:
            Score 0.05-1.0 based on recency band
        """
        if not last_accessed:
            return 0.05

        try:
            last_dt = datetime.fromisoformat(last_accessed.replace('Z', '+00:00'))
            now = datetime.utcnow()
            hours_ago = (now - last_dt.replace(tzinfo=None)).total_seconds() / 3600

            for band in RECENCY_BANDS:
                if hours_ago <= band['maxHours']:
                    return band['score']

            return 0.05
        except Exception:
            return 0.05

    def _frequency_score(self, access_count: int) -> float:
        """
        Compute frequency score (logarithmic, capped at 1.0).

        Args:
            access_count: Number of times node was accessed

        Returns:
            Score 0-1.0: min(1.0, log2(max(1, count)) / 10)
        """
        return min(1.0, math.log2(max(1, access_count)) / 10)

    def _combined_score(self, relevance: float, recency: float, frequency: float,
                       emotion: float, locked: bool) -> float:
        """
        Compute combined relevance score blending all factors.

        Args:
            relevance: Keyword/graph relevance (0-1)
            recency: Recency score (0-1)
            frequency: Frequency score (0-1)
            emotion: Emotional intensity (0-1)
            locked: Whether node is locked (boosts relevance)

        Returns:
            Combined score (0-1)
        """
        # Emotion intensity: raw emotion (0-1) with floor so neutral isn't zero
        emotion_score = min(1.0, EMOTION_FLOOR + (emotion or 0) * (1 - EMOTION_FLOOR))

        if locked:
            # Locked nodes: relevance dominant, emotion still matters
            return relevance * 0.5 + recency * 0.2 + frequency * 0.05 + emotion_score * 0.25

        # Normal blend by weights
        return (relevance * RELEVANCE_WEIGHT +
                recency * RECENCY_WEIGHT +
                frequency * FREQUENCY_WEIGHT +
                emotion_score * EMOTION_WEIGHT)

    def _classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent and extract metadata (type boosts, temporal filter, etc).

        Args:
            query: User query string

        Returns:
            Dict with:
                - intent: 'general' or specific intent name
                - typeBoosts: Dict of type→boost multipliers
                - temporalFilter: {'after': ISO, 'before': ISO} or None
                - followEdges: bool for deeper edge traversal
        """
        lower_query = query.lower()
        intent = 'general'
        type_boosts = {}
        temporal_filter = None
        follow_edges = False

        # Check intent patterns
        for intent_name, pattern in INTENT_PATTERNS.items():
            if pattern.search(lower_query):
                intent = intent_name
                type_boosts = INTENT_TYPE_BOOSTS.get(intent_name, {}).copy()
                break

        # Reasoning chains need deeper edge traversal
        if intent == 'reasoning_chain':
            follow_edges = True

        # Check temporal patterns
        for temporal in TEMPORAL_PATTERNS:
            pattern = temporal['pattern']
            match = pattern.search(lower_query)
            if match:
                range_fn = temporal['range_fn']
                try:
                    temporal_filter = range_fn(match)
                except TypeError:
                    temporal_filter = range_fn()
                if intent == 'general':
                    intent = 'temporal'
                type_boosts.update(INTENT_TYPE_BOOSTS.get('temporal', {}))
                break

        return {
            'intent': intent,
            'typeBoosts': type_boosts,
            'temporalFilter': temporal_filter,
            'followEdges': follow_edges
        }

    # ─── TF-IDF Methods ───

    def _tfidf_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for TF-IDF: expand CamelCase, lowercase, remove stopwords.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (length > 2, non-stopword)
        """
        if not text:
            return []

        # Split CamelCase before lowercasing: "UserDashboard" → "User Dashboard"
        expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        expanded = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', expanded)

        # Lowercase, remove non-alphanumeric (keep hyphens, dots), split
        tokens = expanded.lower()
        tokens = re.sub(r'[^a-z0-9\s\-\.]', ' ', tokens)
        tokens = re.split(r'[\s\-\.]+', tokens)

        # Filter: length > 2, not stopword, remove trailing non-alphanumeric
        result = []
        for w in tokens:
            w = re.sub(r'[^a-z0-9]', '', w)
            if len(w) > 2 and w not in TFIDF_STOP_WORDS:
                result.append(w)

        return result

    def _compute_tf(self, text: str) -> Dict[str, float]:
        """
        Compute term frequency vector (augmented TF formula).

        Args:
            text: Text to analyze

        Returns:
            Dict of term→TF value (0-1)
        """
        tokens = self._tfidf_tokenize(text)
        if not tokens:
            return {}

        # Count term frequencies
        freq = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Augmented TF: 0.5 + 0.5 * (count / max_freq)
        max_freq = max(freq.values()) if freq else 1
        tf = {}
        for term, count in freq.items():
            tf[term] = 0.5 + 0.5 * (count / max_freq)

        return tf

    def _store_tfidf_vector(self, node_id: str, title: str, content: Optional[str], keywords: Optional[str]):
        """
        Store TF-IDF vector for a node (title + content + keywords).

        Args:
            node_id: Node ID
            title: Node title
            content: Node content (optional)
            keywords: Node keywords (optional)
        """
        full_text = ' '.join(filter(None, [title, content, keywords]))
        tf = self._compute_tf(full_text)

        # Delete old vectors for this node
        self.conn.execute('DELETE FROM node_vectors WHERE node_id = ?', (node_id,))

        # Update document frequency counts
        for term in tf.keys():
            self.conn.execute(
                'INSERT INTO doc_freq (term, count) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET count = count + 1',
                (term,)
            )

        # Store TF values (TF-IDF computed at query time)
        for term, tf_val in tf.items():
            self.conn.execute(
                'INSERT OR REPLACE INTO node_vectors (node_id, term, tf) VALUES (?, ?, ?)',
                (node_id, term, tf_val)
            )

        self.conn.commit()

    def _tfidf_score(self, query_terms: List[str], node_id: str) -> float:
        """
        Compute TF-IDF cosine similarity between query and single node.

        Args:
            query_terms: Tokenized query
            node_id: Node to score

        Returns:
            Cosine similarity (0-1)
        """
        if not query_terms:
            return 0

        total_docs = self._get_node_count()
        if total_docs == 0:
            return 0

        # Build query vector
        query_vec = {}
        for term in query_terms:
            query_vec[term] = query_vec.get(term, 0) + 1

        # Normalize query vector
        q_max = max(query_vec.values()) if query_vec else 1
        for t in query_vec:
            query_vec[t] /= q_max

        # Get node's TF values for matching terms
        placeholders = ','.join('?' * len(query_terms))
        cursor = self.conn.execute(
            f'SELECT term, tf FROM node_vectors WHERE node_id = ? AND term IN ({placeholders})',
            [node_id] + query_terms
        )
        node_terms = {row[0]: row[1] for row in cursor.fetchall()}

        if not node_terms:
            return 0

        # Compute cosine similarity with IDF weighting
        dot_product = 0
        query_norm = 0
        doc_norm = 0

        for term in set(list(query_vec.keys()) + list(node_terms.keys())):
            # IDF = log(N / df)
            cursor = self.conn.execute('SELECT count FROM doc_freq WHERE term = ?', (term,))
            row = cursor.fetchone()
            df = row[0] if row else 1
            idf = math.log((total_docs + 1) / (df + 1)) + 1  # smoothed IDF

            q_val = (query_vec.get(term, 0) or 0) * idf
            d_val = (node_terms.get(term, 0) or 0) * idf

            dot_product += q_val * d_val
            query_norm += q_val * q_val
            doc_norm += d_val * d_val

        denom = math.sqrt(query_norm) * math.sqrt(doc_norm)
        return dot_product / denom if denom > 0 else 0

    def _batch_tfidf_scores(self, query_terms: List[str], node_ids: List[str]) -> Dict[str, float]:
        """
        Batch compute TF-IDF scores for multiple nodes (efficient).

        Args:
            query_terms: Tokenized query
            node_ids: List of node IDs to score

        Returns:
            Dict of node_id→score
        """
        if not query_terms or not node_ids:
            return {}

        total_docs = self._get_node_count()
        if total_docs == 0:
            return {}

        # Precompute IDF for all query terms
        idf_map = {}
        for term in set(query_terms):
            cursor = self.conn.execute('SELECT count FROM doc_freq WHERE term = ?', (term,))
            row = cursor.fetchone()
            df = row[0] if row else 1
            idf_map[term] = math.log((total_docs + 1) / (df + 1)) + 1

        # Build query vector
        query_vec = {}
        for term in query_terms:
            query_vec[term] = query_vec.get(term, 0) + 1

        q_max = max(query_vec.values()) if query_vec else 1
        for t in query_vec:
            query_vec[t] /= q_max

        # Query norm (constant for all docs)
        query_norm_sq = 0
        for term, q_val in query_vec.items():
            idf = idf_map.get(term, 1)
            query_norm_sq += (q_val * idf) ** 2

        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0:
            return {}

        # Get all matching vectors in one query
        unique_terms = list(set(query_terms))
        term_placeholders = ','.join('?' * len(unique_terms))
        node_placeholders = ','.join('?' * len(node_ids))
        cursor = self.conn.execute(
            f'SELECT node_id, term, tf FROM node_vectors WHERE term IN ({term_placeholders}) AND node_id IN ({node_placeholders})',
            unique_terms + node_ids
        )

        # Group by node_id
        node_term_maps = {}
        for node_id, term, tf in cursor.fetchall():
            if node_id not in node_term_maps:
                node_term_maps[node_id] = {}
            node_term_maps[node_id][term] = tf

        # Compute similarity for each node
        scores = {}
        for node_id in node_ids:
            node_term_map = node_term_maps.get(node_id)
            if not node_term_map:
                scores[node_id] = 0
                continue

            dot_product = 0
            doc_norm_sq = 0

            for term, tf_val in node_term_map.items():
                idf = idf_map.get(term, 1)
                d_val = tf_val * idf
                q_val = (query_vec.get(term, 0) or 0) * idf
                dot_product += q_val * d_val
                doc_norm_sq += d_val * d_val

            doc_norm = math.sqrt(doc_norm_sq)
            scores[node_id] = dot_product / (query_norm * doc_norm) if doc_norm > 0 else 0

        return scores

    def _rebuild_tfidf_index(self):
        """Rebuild TF-IDF index for all existing (non-archived) nodes."""
        # Clear existing index
        self.conn.execute('DELETE FROM node_vectors')
        self.conn.execute('DELETE FROM doc_freq')

        # Fetch all non-archived nodes
        cursor = self.conn.execute('SELECT id, title, content, keywords FROM nodes WHERE archived = 0')
        all_nodes = cursor.fetchall()

        for node_id, title, content, keywords in all_nodes:
            full_text = ' '.join(filter(None, [title, content, keywords]))
            tf = self._compute_tf(full_text)

            # Update doc_freq
            for term in tf.keys():
                self.conn.execute(
                    'INSERT INTO doc_freq (term, count) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET count = count + 1',
                    (term,)
                )

            # Store TF values
            for term, tf_val in tf.items():
                self.conn.execute(
                    'INSERT OR REPLACE INTO node_vectors (node_id, term, tf) VALUES (?, ?, ?)',
                    (node_id, term, tf_val)
                )

        self.conn.commit()

    # ─── Connection/Edge Management ───

    def connect(self, source_id: str, target_id: str, relation: str = 'related', weight: float = 0.5):
        """
        Create or strengthen an edge between two nodes.
        Bidirectional Hebbian learning.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation type (e.g., 'related', 'inspired_by')
            weight: Edge weight (0-1)
        """
        ts = self.now()

        # Check if edge already exists
        cursor = self.conn.execute(
            'SELECT weight, co_access_count FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        )
        existing = cursor.fetchone()

        if existing:
            old_weight, count = existing
            new_weight = min(MAX_WEIGHT, old_weight + LEARNING_RATE * 0.5)
            self.conn.execute(
                'UPDATE edges SET weight = ?, co_access_count = ?, last_strengthened = ?, relation = ? WHERE source_id = ? AND target_id = ?',
                (new_weight, count + 1, ts, relation, source_id, target_id)
            )
        else:
            # Create bidirectional edge
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, 1, 1.0, ?, ?)',
                (source_id, target_id, weight, relation, ts, ts)
            )
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, 1, 1.0, ?, ?)',
                (target_id, source_id, weight, relation, ts, ts)
            )

        self.conn.commit()

    def connect_typed(self, source_id: str, target_id: str, relation: str = 'related',
                     weight: Optional[float] = None, edge_type: Optional[str] = None,
                     description: str = ''):
        """
        Create or strengthen typed edge with description.
        Edge types can be any string; EDGE_TYPES defines decay behavior for known types.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation: Relation name
            weight: Edge weight (optional; uses EDGE_TYPES default if not provided)
            edge_type: Edge type (optional; defaults to relation)
            description: Human-readable description of connection
        """
        if not edge_type:
            edge_type = relation

        # Known types get configured weight; unknown types get 0.5 default
        edge_def = EDGE_TYPES.get(edge_type)
        actual_weight = weight if weight is not None else (edge_def.get('defaultWeight', 0.5) if edge_def else 0.5)

        ts = self.now()

        # Check if edge already exists
        cursor = self.conn.execute(
            'SELECT weight, co_access_count FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        )
        existing = cursor.fetchone()

        if existing:
            old_weight, count = existing
            new_weight = min(MAX_WEIGHT, old_weight + LEARNING_RATE * 0.5)
            self.conn.execute(
                'UPDATE edges SET weight = ?, co_access_count = ?, last_strengthened = ?, relation = ?, edge_type = ?, description = ? WHERE source_id = ? AND target_id = ?',
                (new_weight, count + 1, ts, relation, edge_type, description, source_id, target_id)
            )
        else:
            # Create bidirectional edge
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, edge_type, description, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
                (source_id, target_id, actual_weight, relation, edge_type, description, ts, ts)
            )
            self.conn.execute(
                'INSERT OR IGNORE INTO edges (source_id, target_id, weight, relation, edge_type, description, co_access_count, stability, last_strengthened, created_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1.0, ?, ?)',
                (target_id, source_id, actual_weight, relation, edge_type, description, ts, ts)
            )

        self.conn.commit()

    # ─── Embedding Integration ───

    async def store_embedding(self, node_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Embed text via embedder and store as BLOB in node_embeddings table.
        Fire-and-forget async; non-critical if fails.

        Args:
            node_id: Node ID
            text: Text to embed

        Returns:
            {'node_id': str, 'embed_ms': int} or None on failure
        """
        if not embedder.is_ready():
            return None

        t0 = time.time()
        blob = embedder.embed(text)  # Already returns bytes
        if not blob:
            return None

        try:
            self.conn.execute(
                'INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
                (node_id, blob, embedder.MODEL_NAME, self.now())
            )
            self.conn.commit()

            return {'node_id': node_id, 'embed_ms': int((time.time() - t0) * 1000)}
        except Exception as e:
            print(f'[brain] Failed to store embedding for {node_id}: {e}')
            return None

    def semantic_recall(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Pure embedding-based search (brute-force cosine scan).
        Embed query, compute cosine similarity against all stored embeddings.

        Args:
            query: Query text
            limit: Max results

        Returns:
            List of {'id': str, 'similarity': float} dicts, sorted by similarity
        """
        if not embedder.is_ready():
            return []

        t0 = time.time()
        query_vec = embedder.embed(query)
        if not query_vec:
            return []

        # Load all embeddings (excluding archived nodes)
        cursor = self.conn.execute(
            'SELECT ne.node_id, ne.embedding FROM node_embeddings ne JOIN nodes n ON n.id = ne.node_id WHERE n.archived = 0'
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Score every node
        scored = []
        for node_id, blob in rows:
            if not blob:
                continue
            similarity = embedder.cosine_similarity(query_vec, blob)
            scored.append({'id': node_id, 'similarity': similarity})

        # Sort and take top-k
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        return scored[:limit]

    def backfill_embeddings(self, batch_size: int = 20) -> int:
        """
        Backfill embeddings for nodes missing them.
        Runs during consolidation; picks recently-accessed nodes first.

        Args:
            batch_size: Max nodes to embed in this batch

        Returns:
            Number of embeddings stored
        """
        if not embedder.is_ready():
            return 0

        # Find up to batch_size nodes without embeddings (order by last_accessed DESC)
        cursor = self.conn.execute(
            '''SELECT n.id, n.title, n.content FROM nodes n
               LEFT JOIN node_embeddings ne ON ne.node_id = n.id
               WHERE ne.node_id IS NULL AND n.archived = 0
               ORDER BY n.last_accessed DESC
               LIMIT ?''',
            (batch_size,)
        )
        nodes = cursor.fetchall()

        if not nodes:
            return 0

        # Build embed texts: title + content (same as store_embedding)
        texts = [f'{title}{(" " + content) if content else ""}' for _, title, content in nodes]

        # Batch embed
        embeddings = embedder.embed_batch(texts)
        stored = 0

        for i, (node_id, _, _) in enumerate(nodes):
            if i >= len(embeddings) or not embeddings[i]:
                continue  # Skip failed individual embeds

            blob = embeddings[i]  # Already bytes from embed_batch
            try:
                self.conn.execute(
                    'INSERT OR REPLACE INTO node_embeddings (node_id, embedding, model, created_at) VALUES (?, ?, ?, ?)',
                    (node_id, blob, embedder.MODEL_NAME, self.now())
                )
                stored += 1
            except Exception:
                pass  # Skip failed; will retry next cycle

        self.conn.commit()
        return stored

    # ─── Session Activity Tracking ───

    def _get_session_activity(self) -> Dict[str, Any]:
        """
        Read session activity from brain_meta table.

        Returns:
            Dict with remember_count, edit_check_count, etc.
        """
        try:
            cursor = self.conn.execute(
                'SELECT key, value FROM brain_meta WHERE key IN (?, ?, ?)',
                ('remember_count', 'edit_check_count', 'session_id')
            )
            result = {}
            for key, value in cursor.fetchall():
                if key.endswith('_count'):
                    result[key] = int(value) if value else 0
                else:
                    result[key] = value
            return result
        except Exception:
            return {}

    def _update_session_activity(self, key: str, value: Any):
        """
        Write session activity to brain_meta table.

        Args:
            key: Config key
            value: Config value (will be stringified)
        """
        self.conn.execute(
            'INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES (?, ?, ?)',
            (key, str(value), self.now())
        )
        self.conn.commit()

    def reset_session_activity(self):
        """Reset session counters for new session."""
        self._update_session_activity('remember_count', 0)
        self._update_session_activity('edit_check_count', 0)
        self._update_session_activity('session_id', uuid.uuid4().hex)

    def record_remember(self):
        """Increment remember counter."""
        activity = self._get_session_activity()
        count = activity.get('remember_count', 0) + 1
        self._update_session_activity('remember_count', count)

    def record_edit_check(self):
        """Increment edit check counter."""
        activity = self._get_session_activity()
        count = activity.get('edit_check_count', 0) + 1
        self._update_session_activity('edit_check_count', count)

    # ─── Utilities ───

    def _get_node_count(self) -> int:
        """Get count of non-archived nodes."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM nodes WHERE archived = 0')
        return cursor.fetchone()[0]

    def _get_edge_count(self) -> int:
        """Get total edge count."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM edges')
        return cursor.fetchone()[0]

    def _get_locked_count(self) -> int:
        """Get count of locked nodes."""
        cursor = self.conn.execute('SELECT COUNT(*) FROM nodes WHERE locked = 1 AND archived = 0')
        return cursor.fetchone()[0]

    # ─── REMEMBER: Store a new node with TF-IDF + embeddings ───

    def remember(self, type: str, title: str, content: Optional[str] = None,
                 keywords: Optional[str] = None, locked: bool = False,
                 connections: Optional[List[Dict[str, Any]]] = None,
                 emotion: float = 0, emotion_label: str = 'neutral',
                 emotion_source: str = 'auto', project: Optional[str] = None,
                 confidence: float = 1.0) -> Dict[str, Any]:
        """
        Store a new memory node with semantic indexing and connections.

        Args:
            type: Node type (person, project, task, etc.)
            title: Node title
            content: Optional detailed content
            keywords: Optional keywords (auto-extracted if not provided)
            locked: If True, node never decays
            connections: List of {target_id, relation, weight} dicts
            emotion: Emotional intensity (0-1)
            emotion_label: 'positive', 'negative', 'neutral', 'frustration', etc.
            emotion_source: 'auto', 'user', 'system'
            project: Optional project ID to associate
            confidence: Confidence score (for future use)

        Returns:
            Dict with id, type, title, emotion, emotion_label, bridges_created
        """
        node_id = self._generate_id(type)
        ts = self.now()

        # Extract keywords if not provided
        if not keywords:
            keywords = self._extract_keywords(f'{title} {content or ""}')

        # INSERT into nodes table
        self.conn.execute(
            '''INSERT INTO nodes
               (id, type, title, content, keywords, activation, stability, locked,
                recency_score, emotion, emotion_label, emotion_source, project,
                last_accessed, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 1.0, 1.0, ?, 1.0, ?, ?, ?, ?, ?, ?, ?)''',
            (node_id, type, title, content, keywords, 1 if locked else 0,
             emotion, emotion_label, emotion_source, project, ts, ts, ts)
        )
        self.conn.commit()

        # v5: Build TF-IDF vector for this node
        try:
            self._store_tfidf_vector(node_id, title, content, keywords)
        except Exception:
            pass  # Non-critical — recall still works via keywords

        # v13: Store dense embedding via fire-and-forget async
        embed_text = f'{title}{" " + content if content else ""}'

        def store_embed():
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.store_embedding(node_id, embed_text))
            except Exception:
                pass

        # Fire-and-forget: start embedding in background
        thread = threading.Thread(target=store_embed, daemon=True)
        thread.start()

        # Create connections
        if connections:
            for conn in connections:
                target_id = conn.get('target_id')
                relation = conn.get('relation', 'related')
                weight = conn.get('weight', 0.5)
                if target_id:
                    self.connect(node_id, target_id, relation, weight)

        # v11: Emergent bridging at store-time
        bridges = []
        try:
            bridges = self._bridge_at_store_time(node_id)
        except Exception:
            pass  # Non-critical — bridging failure should never block remember

        return {
            'id': node_id,
            'type': type,
            'title': title,
            'emotion': emotion,
            'emotion_label': emotion_label,
            'bridges_created': len(bridges)
        }

    # ─── RECALL: v5 with TF-IDF + intent detection + temporal filtering + decay ───

    def recall(self, query: str, types: Optional[List[str]] = None, limit: int = 20,
               offset: int = 0, include_archived: bool = False, min_recency: float = 0,
               project: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve relevant nodes with TF-IDF scoring, spreading activation, and decay.

        Args:
            query: Search query
            types: Filter by node types
            limit: Max results to return
            offset: Pagination offset
            include_archived: Include archived nodes
            min_recency: Minimum recency score threshold
            project: Filter to specific project
            session_id: Optional session ID for logging

        Returns:
            Dict with results (list of nodes), _recall_log_id, intent
        """
        limit = min(limit, MAX_PAGE_SIZE)

        # v5 Step 0: Intent detection
        intent_data = self._classify_intent(query)
        intent = intent_data['intent']
        type_boosts = intent_data['typeBoosts']
        temporal_filter = intent_data['temporalFilter']

        # Step 1: Keyword search for seeds
        seeds = self._search_keywords(query, 30)

        all_seeds = {}
        for seed in seeds:
            if seed['id'] not in all_seeds:
                all_seeds[seed['id']] = seed

        # v5: Also find seeds via TF-IDF
        tfidf_query_terms = self._tfidf_tokenize(query)
        if tfidf_query_terms:
            try:
                unique_terms = list(set(tfidf_query_terms))
                placeholders = ','.join('?' * len(unique_terms))
                cursor = self.conn.execute(
                    f'''SELECT DISTINCT nv.node_id FROM node_vectors nv
                        JOIN nodes n ON n.id = nv.node_id
                        WHERE nv.term IN ({placeholders}) AND n.archived = 0
                        LIMIT 50''',
                    unique_terms
                )
                for row in cursor.fetchall():
                    nid = row[0]
                    if nid not in all_seeds:
                        # Fetch node data
                        cursor2 = self.conn.execute(
                            '''SELECT id, type, title, content, keywords, activation, stability,
                                      access_count, locked, archived, last_accessed, created_at
                               FROM nodes WHERE id = ?''',
                            (nid,)
                        )
                        row2 = cursor2.fetchone()
                        if row2:
                            all_seeds[nid] = {
                                'id': row2[0], 'type': row2[1], 'title': row2[2],
                                'content': row2[3], 'keywords': row2[4],
                                'activation': row2[5], 'stability': row2[6],
                                'access_count': row2[7], 'locked': row2[8] == 1,
                                'archived': row2[9] == 1, 'last_accessed': row2[10],
                                'created_at': row2[11]
                            }
            except Exception:
                pass

        if not all_seeds:
            # Return recent nodes if no seeds found
            return {
                'results': self._get_recent(limit, types),
                '_recall_log_id': None,
                'intent': intent
            }

        # Step 1b: Compute direct keyword match strength per seed
        query_terms = [w.replace('[^a-z0-9]', '', ) for w in query.lower().split()
                       if len(w) > 2]
        query_terms = [w for w in query_terms if w]

        direct_match_scores = {}
        for seed_id, seed in all_seeds.items():
            kw = (seed.get('keywords') or '').lower()
            title = (seed.get('title') or '').lower()
            content = (seed.get('content') or '').lower()
            match_count = 0
            for term in query_terms:
                if term in kw or term in title or term in content:
                    match_count += 1
            direct_match_scores[seed_id] = (match_count / len(query_terms)) if query_terms else 0

        # Step 2: Spreading activation
        activated = self.spread_activation(list(all_seeds.keys()), types)

        # v5: Compute batch TF-IDF scores
        activated_ids = [n['id'] for n in activated]
        tfidf_scores = self._batch_tfidf_scores(tfidf_query_terms, activated_ids)

        # Step 3: Compute combined score with TF-IDF + keyword + intent boosts
        max_spread = max([n.get('spread_activation', 0.001) for n in activated] or [0.001])

        now_ms = time.time() * 1000  # milliseconds

        scored = []
        for node in activated:
            # Keyword-based relevance
            keyword_relevance = node.get('spread_activation', 0) / max_spread

            direct_match = direct_match_scores.get(node['id'], 0)
            if direct_match == 0 and query_terms:
                nkw = (node.get('keywords') or '').lower()
                ntitle = (node.get('title') or '').lower()
                ncontent = (node.get('content') or '').lower()
                mc = 0
                for term in query_terms:
                    if term in nkw or term in ntitle or term in ncontent:
                        mc += 1
                direct_match = mc / len(query_terms)

            if direct_match > 0:
                keyword_relevance = min(1.0, keyword_relevance + direct_match * 0.5)

            # v5: TF-IDF semantic relevance
            semantic_score = tfidf_scores.get(node['id'], 0)

            # v5: Blend keyword and semantic scores
            relevance = (TFIDF_KEYWORD_WEIGHT * keyword_relevance +
                        TFIDF_SEMANTIC_WEIGHT * semantic_score)

            # v4: Hub dampening (nodes with 20+ connections get reduced relevance)
            cursor = self.conn.execute(
                'SELECT COUNT(*) FROM edges WHERE source_id = ?',
                (node['id'],)
            )
            edge_count = cursor.fetchone()[0]
            if edge_count > 40:
                relevance *= 40 / edge_count

            # v4: Type dampening
            if node.get('type') in ('project', 'person'):
                relevance *= 0.5

            # v5: Intent-based type boosting
            type_boost = type_boosts.get(node.get('type'), 1.0)
            relevance *= type_boost

            recency = self._recency_score(node.get('last_accessed'))
            frequency = self._frequency_score(node.get('access_count', 0))
            emotion_intensity = abs(node.get('emotion', 0))

            # ─── Ebbinghaus retention with time-dilation ───
            retention = 1.0
            if not node.get('locked'):
                half_life = DECAY_HALF_LIFE.get(node.get('type'), 168)
                if half_life != float('inf'):
                    last_accessed_str = node.get('last_accessed')
                    if last_accessed_str:
                        try:
                            last_accessed_dt = datetime.fromisoformat(last_accessed_str.replace('Z', '+00:00'))
                            last_accessed_ms = last_accessed_dt.timestamp() * 1000
                        except Exception:
                            last_accessed_ms = now_ms

                        wall_clock_hours = (now_ms - last_accessed_ms) / (1000 * 60 * 60)

                        # Read time-dilation rates from brain config
                        active_rate = self.get_config('decay_active_rate', 1.0)
                        idle_rate = self.get_config('decay_idle_rate', 0.1)

                        # Calculate session vs idle hours since access
                        total_session_minutes = float(self.get_config('total_session_minutes', 0) or 0)
                        last_session_minutes = float(self.get_config('_last_session_minutes_at_access', 0) or 0)
                        session_hours_since_access = max(0, (total_session_minutes - last_session_minutes) / 60)
                        idle_hours = max(0, wall_clock_hours - session_hours_since_access)

                        # Dilated time
                        dilated_hours = (session_hours_since_access * active_rate +
                                        idle_hours * idle_rate)

                        effective_s = node.get('stability', 1.0) * half_life
                        retention = math.exp(-dilated_hours / effective_s) if effective_s > 0 else 1.0

                        # Stability floor
                        if (node.get('access_count', 0) >= STABILITY_FLOOR_ACCESS_THRESHOLD and
                            retention < STABILITY_FLOOR_RETENTION):
                            retention = STABILITY_FLOOR_RETENTION

            if emotion_intensity > 0.5:
                retention = min(1.0, retention * (1 + emotion_intensity * 0.5))

            combined = self._combined_score(relevance, recency, frequency,
                                          emotion_intensity, node.get('locked', False))
            effective = combined * retention

            scored.append({
                **node,
                'recency_score': recency,
                'frequency_score': frequency,
                'relevance_score': relevance,
                'semantic_score': semantic_score,
                'keyword_relevance': keyword_relevance,
                'emotion_intensity': emotion_intensity,
                'retention': retention,
                'effective_activation': effective
            })

        # Step 4: Filter
        filtered = scored
        if not include_archived:
            filtered = [n for n in filtered if not n.get('archived')]
        if types:
            filtered = [n for n in filtered if n.get('type') in types]
        if min_recency > 0:
            filtered = [n for n in filtered if n.get('recency_score', 0) >= min_recency]

        # v5: Project filter
        if project:
            filtered.sort(key=lambda n: (1 if n.get('project') == project else 0, -n.get('effective_activation', 0)))

        # v5: Temporal filter
        if temporal_filter:
            after = temporal_filter.get('after')
            before = temporal_filter.get('before')
            filtered = [n for n in filtered if self._matches_temporal_filter(n.get('created_at'), after, before)]

        # Step 5: Sort by effective activation (if no project filter)
        if not project:
            filtered.sort(key=lambda n: -n.get('effective_activation', 0))

        # Step 6: Pagination
        page = filtered[offset:offset + limit]

        # Step 7: Mark accessed + Hebbian
        if not session_id:
            session_id = f'ses_{int(time.time() * 1000)}'

        for node in page:
            self._mark_accessed(node['id'], session_id)

        self._hebbian_strengthen([n['id'] for n in page])

        # v4: Auto-instrument
        returned_ids = [n['id'] for n in page]
        recall_log_id = None
        try:
            recall_log_id = self._log_recall(session_id, query, returned_ids)
        except Exception:
            pass

        # v6: Attach reasoning chains when intent is reasoning_chain
        reasoning_chains = []
        if intent == 'reasoning_chain':
            # 1. Pull chains for decision nodes in results
            decision_nodes = [n for n in page if n.get('type') == 'decision']
            for dn in decision_nodes:
                # Note: reasoning methods not yet implemented, skipping for now
                pass

        result = {
            'results': page,
            '_recall_log_id': recall_log_id,
            'intent': intent,
        }

        if reasoning_chains:
            result['reasoning_chains'] = reasoning_chains

        return result

    # ─── RECALL WITH EMBEDDINGS: Blend TF-IDF with semantic embeddings ───

    def recall_with_embeddings(self, query: str, types: Optional[List[str]] = None,
                                     limit: int = 20, offset: int = 0,
                                     include_archived: bool = False,
                                     min_recency: float = 0, project: Optional[str] = None,
                                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance recall with embedding-based semantic search.

        Combines TF-IDF keyword matching with dense embeddings (bge-m3).
        Graceful degradation if embedder isn't ready.

        Args:
            query: Search query
            types: Filter by node types
            limit: Max results
            offset: Pagination offset
            include_archived: Include archived
            min_recency: Min recency threshold
            project: Optional project filter
            session_id: Optional session ID

        Returns:
            Dict with results, _recall_log_id, _embedding_stats, intent
        """
        t0 = time.time()

        # STEP 1: Standard recall (TF-IDF + keyword)
        result = self.recall(query, types, limit, offset, include_archived,
                           min_recency, project, session_id)

        # Graceful degradation
        if not embedder.is_ready() or not result.get('results'):
            return result

        # STEP 2: Embed the query
        try:
            query_vec = embedder.embed(query)
            if not query_vec:
                return result
        except Exception:
            return result

        # STEP 3: Configure blend weight
        embedding_weight = float(self.get_config('embedding_blend_weight', 0.3) or 0.3)

        # STEP 4: Fetch embeddings for TF-IDF results
        result_ids = [r['id'] for r in result.get('results', [])]
        placeholders = ','.join('?' * len(result_ids)) if result_ids else ''
        embedding_map = {}

        if result_ids:
            try:
                cursor = self.conn.execute(
                    f'''SELECT node_id, embedding FROM node_embeddings
                        WHERE node_id IN ({placeholders})''',
                    result_ids
                )
                for row in cursor.fetchall():
                    node_id = row[0]
                    blob = row[1]
                    try:
                        vec = blob
                        if vec:
                            embedding_map[node_id] = vec
                    except Exception:
                        pass
            except Exception:
                pass

        # STEP 5: Embedding-only discovery (nodes >0.5 cosine similarity not in TF-IDF results)
        embedding_only_hits = []
        try:
            cursor = self.conn.execute(
                '''SELECT ne.node_id, ne.embedding FROM node_embeddings ne
                   JOIN nodes n ON n.id = ne.node_id
                   WHERE n.archived = 0'''
            )
            existing_ids = set(result_ids)
            candidates = []
            for row in cursor.fetchall():
                node_id = row[0]
                if node_id in existing_ids:
                    continue
                blob = row[1]
                try:
                    vec = blob
                    if vec:
                        sim = embedder.cosine_similarity(query_vec, vec)
                        if sim > 0.5:
                            candidates.append((node_id, sim))
                except Exception:
                    pass

            candidates.sort(key=lambda x: -x[1])
            embedding_only_hits = candidates[:5]
        except Exception:
            pass

        # STEP 6: Blend embedding scores into TF-IDF results
        for node in result.get('results', []):
            vec = embedding_map.get(node['id'])
            if vec:
                sim = embedder.cosine_similarity(query_vec, vec)
                node['embedding_similarity'] = round(sim * 1000) / 1000
                node['effective_activation'] = (
                    (1 - embedding_weight) * node['effective_activation'] +
                    embedding_weight * sim
                )
            else:
                node['embedding_similarity'] = None

        # STEP 7: Inject embedding-only discoveries
        for node_id, similarity in embedding_only_hits:
            try:
                cursor = self.conn.execute(
                    '''SELECT id, type, title, content, keywords, activation, stability,
                              access_count, locked, archived, last_accessed, created_at,
                              emotion, emotion_label
                       FROM nodes WHERE id = ?''',
                    (node_id,)
                )
                row = cursor.fetchone()
                if row:
                    result['results'].append({
                        'id': row[0], 'type': row[1], 'title': row[2],
                        'content': row[3], 'keywords': row[4],
                        'activation': row[5], 'stability': row[6],
                        'access_count': row[7], 'locked': row[8] == 1,
                        'archived': row[9] == 1, 'last_accessed': row[10],
                        'created_at': row[11], 'emotion': row[12],
                        'emotion_label': row[13],
                        'embedding_similarity': round(similarity * 1000) / 1000,
                        'effective_activation': similarity * embedding_weight,
                        '_source': 'embedding_only',
                    })
            except Exception:
                pass

        # STEP 8: Re-sort and trim
        result['results'].sort(key=lambda n: -n.get('effective_activation', 0))
        if limit:
            result['results'] = result['results'][:min(limit, MAX_PAGE_SIZE)]

        # STEP 9: Debug metadata
        recall_ms = (time.time() - t0) * 1000
        result['_embedding_stats'] = {
            'blend_weight': embedding_weight,
            'nodes_with_embeddings': len(embedding_map),
            'embedding_only_hits': len(embedding_only_hits),
            'recall_ms': recall_ms,
        }

        return result

    # ─── SPREAD ACTIVATION: Multi-hop semantic activation ───

    def spread_activation(self, seed_ids: List[str], types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Spread activation from seed nodes through graph edges.

        Multi-hop with exponential decay (0.5^hop).
        Each hop: get neighbors, multiply activation by edge_weight * decay.
        MAX_HOPS=3, MAX_NEIGHBORS=50 per node.

        Args:
            seed_ids: Starting node IDs
            types: Optional filter by node types

        Returns:
            List of activated nodes with spread_activation scores
        """
        activation = {}
        node_cache = {}

        for sid in seed_ids:
            activation[sid] = 1.0

        for hop in range(MAX_HOPS):
            decay_factor = SPREAD_DECAY ** (hop + 1)
            current_nodes = [(nid, act) for nid, act in activation.items() if act > 0.01]

            for node_id, node_activation in current_nodes:
                cursor = self.conn.execute(
                    '''SELECT target_id, weight FROM edges
                       WHERE source_id = ? AND weight > ?
                       ORDER BY weight DESC LIMIT ?''',
                    (node_id, PRUNE_THRESHOLD, MAX_NEIGHBORS)
                )

                for row in cursor.fetchall():
                    target_id = row[0]
                    edge_weight = row[1]
                    spread = node_activation * edge_weight * decay_factor
                    current_act = activation.get(target_id, 0)
                    activation[target_id] = current_act + spread

        # Fetch full node data
        results = []
        for node_id, act in activation.items():
            node = node_cache.get(node_id)
            if not node:
                cursor = self.conn.execute(
                    '''SELECT id, type, title, content, keywords, activation, stability,
                              access_count, locked, archived, last_accessed, created_at,
                              emotion, emotion_label, project
                       FROM nodes WHERE id = ?''',
                    (node_id,)
                )
                row = cursor.fetchone()
                if row:
                    node = {
                        'id': row[0], 'type': row[1], 'title': row[2],
                        'content': row[3], 'keywords': row[4],
                        'activation': row[5], 'stability': row[6],
                        'access_count': row[7], 'locked': row[8] == 1,
                        'archived': row[9] == 1, 'last_accessed': row[10],
                        'created_at': row[11],
                        'emotion': row[12] or 0, 'emotion_label': row[13] or 'neutral',
                        'project': row[14]
                    }
                    node_cache[node_id] = node

            if node:
                # Type filter
                if types and node.get('type') not in types:
                    continue

                results.append({**node, 'spread_activation': act})

        return results

    # ─── Helper methods for remember/recall ───

    def _extract_keywords(self, text: str) -> str:
        """
        Extract keywords from text (numbers, proper nouns, technical terms, common words).

        Args:
            text: Text to extract from

        Returns:
            Space-separated keywords string
        """
        if not text:
            return ''

        # PHASE 1: Extract numbers and values before lowercasing
        number_patterns = re.findall(r'\$?\d+(?:\.\d+)?%?(?:px|ms|s|d|kb|mb|gb)?', text, re.IGNORECASE)
        number_keywords = [n.lower().replace(re.sub(r'[^a-z0-9%$.]', '', n), '') for n in number_patterns]
        number_keywords = [n for n in number_keywords if len(n) >= 1]

        # PHASE 2: Extract proper nouns and technical terms
        proper_nouns = re.findall(r'[A-Z][a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*', text)
        technical_terms = re.findall(r'[a-z]+[A-Z][a-zA-Z0-9]*', text)
        snake_terms = re.findall(r'[a-z][a-z0-9]*_[a-z0-9_]+', text)
        dotted_terms = re.findall(r'[a-z]+(?:\.[a-z]+)+', text)

        preserved_terms = set()
        for term in proper_nouns + technical_terms + snake_terms + dotted_terms:
            lower = term.lower()
            if len(lower) > 2 and lower not in TFIDF_STOP_WORDS:
                preserved_terms.add(lower)
                stripped = re.sub(r'[^a-z0-9]', '', lower)
                if len(stripped) > 2 and stripped != lower:
                    preserved_terms.add(stripped)

        # PHASE 3: Standard word extraction
        words = re.sub(r'[^a-z0-9\s\-\./]', ' ', text.lower()).split()
        words = [w for w in words if len(w) > 2 and w not in TFIDF_STOP_WORDS]

        # Also add variants
        variants = set()
        for w in words:
            variants.add(w)
            stripped = re.sub(r'[^a-z0-9]', '', w)
            if stripped != w and len(stripped) > 2:
                variants.add(stripped)

        all_keywords = list(preserved_terms | variants | set(number_keywords))
        return ' '.join(all_keywords[:50])  # Cap at 50 keywords

    def _search_keywords(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search nodes by keyword/title/content.

        Args:
            query: Search query
            limit: Max results

        Returns:
            List of matching nodes
        """
        words = self._tfidf_tokenize(query)
        if not words:
            return []

        # Build OR conditions: (LIKE word1 OR LIKE word2 OR ...)
        conditions = []
        params = []
        for w in words:
            conditions.append('(LOWER(keywords) LIKE ? OR LOWER(title) LIKE ? OR LOWER(content) LIKE ?)')
            params.extend([f'%{w}%', f'%{w}%', f'%{w}%'])

        where_clause = ' OR '.join(conditions)
        params.append(limit)

        try:
            cursor = self.conn.execute(
                f'''SELECT id, type, title, content, keywords, activation, stability,
                           access_count, locked, archived, last_accessed, created_at, project
                    FROM nodes WHERE archived = 0 AND ({where_clause}) LIMIT ?''',
                params
            )
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0], 'type': row[1], 'title': row[2],
                    'content': row[3], 'keywords': row[4],
                    'activation': row[5], 'stability': row[6],
                    'access_count': row[7], 'locked': row[8] == 1,
                    'archived': row[9] == 1, 'last_accessed': row[10],
                    'created_at': row[11], 'project': row[12]
                })
            return results
        except Exception:
            return []

    def _mark_accessed(self, node_id: str, session_id: str):
        """Mark a node as accessed and log it."""
        ts = self.now()
        self.conn.execute(
            '''UPDATE nodes
               SET access_count = access_count + 1,
                   activation = MIN(1.0, activation + 0.1),
                   recency_score = 1.0,
                   last_accessed = ?,
                   updated_at = ?
               WHERE id = ?''',
            (ts, ts, node_id)
        )
        self.conn.execute(
            'INSERT INTO access_log (session_id, node_id, timestamp) VALUES (?, ?, ?)',
            (session_id, node_id, ts)
        )
        self.conn.commit()

    def _hebbian_strengthen(self, node_ids: List[str]):
        """
        Strengthen connections between co-accessed nodes (Hebbian learning).

        If two nodes are co-recalled but have no edge, CREATE a co_accessed edge.
        If they already have an edge, strengthen it.
        This is how the brain auto-discovers relationships from usage patterns.
        """
        if len(node_ids) < 2:
            return

        ts = self.now()

        # Cap pairwise work: only top N nodes to avoid O(n^2) explosion
        ids = node_ids[:15]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                nid_i = ids[i]
                nid_j = ids[j]

                # Check if edge exists (either direction)
                cursor = self.conn.execute(
                    'SELECT weight, co_access_count, stability FROM edges WHERE source_id = ? AND target_id = ?',
                    (nid_i, nid_j)
                )
                row = cursor.fetchone()

                if not row:
                    # Check reverse direction
                    cursor = self.conn.execute(
                        'SELECT weight, co_access_count, stability FROM edges WHERE source_id = ? AND target_id = ?',
                        (nid_j, nid_i)
                    )
                    row = cursor.fetchone()
                    if row:
                        # Reverse exists — update it
                        w, count, stab = row
                        new_weight = min(MAX_WEIGHT, w + LEARNING_RATE * 0.1)
                        new_stability = min(stab * STABILITY_BOOST, 10.0)
                        self.conn.execute(
                            '''UPDATE edges
                               SET weight = ?, co_access_count = ?, stability = ?, last_strengthened = ?
                               WHERE source_id = ? AND target_id = ?''',
                            (new_weight, count + 1, new_stability, ts, nid_j, nid_i)
                        )
                        continue

                if row:
                    # Edge exists — strengthen it
                    w, count, stab = row
                    new_weight = min(MAX_WEIGHT, w + LEARNING_RATE * 0.1)
                    new_stability = min(stab * STABILITY_BOOST, 10.0)

                    self.conn.execute(
                        '''UPDATE edges
                           SET weight = ?, co_access_count = ?, stability = ?, last_strengthened = ?
                           WHERE source_id = ? AND target_id = ?''',
                        (new_weight, count + 1, new_stability, ts, nid_i, nid_j)
                    )
                else:
                    # NO edge exists — CREATE a co_accessed edge
                    # Start with low weight; repeated co-access will strengthen it
                    self.conn.execute(
                        '''INSERT OR IGNORE INTO edges
                           (source_id, target_id, weight, relation, edge_type, co_access_count,
                            stability, last_strengthened, created_at)
                           VALUES (?, ?, ?, 'co_accessed', 'co_accessed', 1, 1.0, ?, ?)''',
                        (nid_i, nid_j, EDGE_TYPES['co_accessed']['defaultWeight'], ts, ts)
                    )

        self.conn.commit()

    def _log_recall(self, session_id: str, query: str, returned_ids: List[str]) -> Optional[str]:
        """Log a recall event."""
        ts = self.now()
        cursor = self.conn.execute(
            '''INSERT INTO recall_log (session_id, query, returned_ids, returned_count, created_at)
               VALUES (?, ?, ?, ?, ?)''',
            (session_id or 'unknown', query, json.dumps(returned_ids), len(returned_ids), ts)
        )
        self.conn.commit()

        # Return the last inserted row ID
        cursor = self.conn.execute('SELECT last_insert_rowid()')
        row = cursor.fetchone()
        return str(row[0]) if row else None

    def _get_recent(self, limit: int = 20, types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get recently accessed nodes."""
        sql = 'SELECT id, type, title, content, keywords, activation, stability, access_count, locked, archived, last_accessed, created_at FROM nodes WHERE archived = 0'
        params = []

        if types:
            placeholders = ','.join('?' * len(types))
            sql += f' AND type IN ({placeholders})'
            params.extend(types)

        sql += ' ORDER BY last_accessed DESC LIMIT ?'
        params.append(limit)

        results = []
        cursor = self.conn.execute(sql, params)
        for row in cursor.fetchall():
            results.append({
                'id': row[0], 'type': row[1], 'title': row[2],
                'content': row[3], 'keywords': row[4],
                'activation': row[5], 'stability': row[6],
                'access_count': row[7], 'locked': row[8] == 1,
                'archived': row[9] == 1, 'last_accessed': row[10],
                'created_at': row[11],
                'spread_activation': row[5],
                'effective_activation': row[5]
            })
        return results

    def _matches_temporal_filter(self, created_at: Optional[str], after: Optional[str], before: Optional[str]) -> bool:
        """Check if a node creation date matches temporal filter."""
        if not created_at:
            return False
        if after and created_at < after:
            return False
        if before and created_at > before:
            return False
        return True

    def _bridge_at_store_time(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Detect bridge opportunities at store-time.
        Returns array of bridges created.
        """
        max_bridges = self.get_config('bridge_max_per_remember', 2)
        candidates = self._find_bridge_candidates(node_id, limit=max_bridges)
        created = []

        for c in candidates:
            bridge = self._create_bridge(node_id, c['targetId'], c.get('sharedTitles', ''))
            if bridge:
                created.append(bridge)

        return created

    def dream(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Spontaneous association: random walks through graph to find interesting connections.
        Seed selection weighted by: connection_count × (1 + emotion) × recency_multiplier.
        Returns dict with dreams, surfaced_insights, bridge_proposals.
        """
        ts = self.now()
        if session_id is None:
            session_id = f'dream_{int(time.time() * 1000)}'

        dreams = []

        # Get pool of candidate seeds (40 random unlocked nodes)
        seed_pool = self.conn.execute('''
            SELECT n.id, n.emotion, n.last_accessed,
                   (SELECT COUNT(*) FROM edges e WHERE e.source_id = n.id AND e.weight >= 0.1) as edge_count
            FROM nodes n
            WHERE n.archived = 0 AND n.locked = 0
            ORDER BY RANDOM()
            LIMIT 40
        ''').fetchall()

        if len(seed_pool) < 2:
            return {'dreams': [], 'message': 'Not enough nodes to dream'}

        # Weighted seed selection
        now = time.time() * 1000  # milliseconds
        seed_candidates = []
        for r in seed_pool:
            node_id, emotion, last_accessed, edge_count = r
            emotion = emotion or 0
            edge_count = edge_count or 0

            # Hours since last access → recency multiplier
            if last_accessed:
                try:
                    last_ts = datetime.fromisoformat(last_accessed.replace('Z', '+00:00')).timestamp() * 1000
                except:
                    last_ts = 0
                hours_ago = max(0, (now - last_ts) / (1000 * 60 * 60))
            else:
                hours_ago = 720  # never accessed = 1 month

            recency_boost = 1.0 + 1.0 / (1 + hours_ago / 24)

            weight = math.sqrt(edge_count + 1) * (1 + emotion * 0.5) * recency_boost
            seed_candidates.append({
                'id': node_id,
                'emotion': emotion,
                'edgeCount': edge_count,
                'recencyBoost': round(recency_boost * 100) / 100,
                'weight': weight
            })

        total_seed_weight = sum(c['weight'] for c in seed_candidates)
        if total_seed_weight <= 0:
            return {'dreams': [], 'message': 'Seed candidates have no weight'}

        def pick_weighted_seed(exclude=None):
            roll = random.random() * total_seed_weight
            for c in seed_candidates:
                if c['id'] == exclude:
                    continue
                roll -= c['weight']
                if roll <= 0:
                    return c
            # Fallback
            for c in seed_candidates:
                if c['id'] != exclude:
                    return c
            return seed_candidates[0]

        # Generate dreams
        for d in range(DREAM_COUNT):
            seed_a_data = pick_weighted_seed()
            seed_a = seed_a_data['id']
            seed_b_data = pick_weighted_seed(seed_a)
            seed_b = seed_b_data['id']
            if seed_b == seed_a:
                continue

            # Random walks
            walk_a = self._random_walk(seed_a, DREAM_WALK_LENGTH)
            walk_b = self._random_walk(seed_b, DREAM_WALK_LENGTH)

            # Endpoints
            end_a = walk_a[-1] if walk_a else seed_a
            end_b = walk_b[-1] if walk_b else seed_b

            # Get titles
            node_a = self._get_node_title(seed_a)
            node_end_a = self._get_node_title(end_a)
            node_b = self._get_node_title(seed_b)
            node_end_b = self._get_node_title(end_b)

            insight = f'Association: "{node_a}" → "{node_end_a}" | "{node_b}" → "{node_end_b}"'

            # Check if edge exists between endpoints
            existing_edge = self.conn.execute(
                'SELECT weight FROM edges WHERE source_id = ? AND target_id = ?',
                (end_a, end_b)
            ).fetchone()

            intuition_id = None
            if not existing_edge and end_a != end_b:
                # Create intuition node
                result = self.remember(
                    type='intuition',
                    title=f'Dream: {node_end_a[:40] if node_end_a else "?"} ↔ {node_end_b[:40] if node_end_b else "?"}',
                    content=insight,
                    keywords=f'dream intuition association {self._extract_keywords(node_end_a + " " + node_end_b)}',
                    locked=False,
                    emotion=0.2,
                    emotion_label='curiosity',
                    emotion_source='dream',
                    connections=[
                        {'target_id': end_a, 'relation': 'dreamed_from', 'weight': 0.3},
                        {'target_id': end_b, 'relation': 'dreamed_from', 'weight': 0.3}
                    ]
                )
                intuition_id = result['id']

            # Log dream
            self.conn.execute('''
                INSERT INTO dream_log (session_id, intuition_node_id, seed_nodes, walk_path, insight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, intuition_id, json.dumps([seed_a, seed_b]),
                  json.dumps([walk_a, walk_b]), insight, ts))

            # Score dream for interestingness
            interest_score = 0

            # Recency check
            try:
                recency_check = self.conn.execute('''
                    SELECT MAX(CASE WHEN julianday('now') - julianday(last_accessed) < 1 THEN 3
                                    WHEN julianday('now') - julianday(last_accessed) < 3 THEN 2
                                    WHEN julianday('now') - julianday(last_accessed) < 7 THEN 1
                                    ELSE 0 END) as recency
                    FROM nodes WHERE id IN (?, ?) AND archived = 0
                ''', (end_a, end_b)).fetchone()
                if recency_check and recency_check[0]:
                    interest_score += recency_check[0]
            except:
                pass

            # Uniqueness check
            try:
                type_check = self.conn.execute(
                    'SELECT type, project FROM nodes WHERE id IN (?, ?)',
                    (end_a, end_b)
                ).fetchall()
                if len(type_check) >= 2:
                    type_a, proj_a = type_check[0][0], type_check[0][1]
                    type_b, proj_b = type_check[1][0], type_check[1][1]
                    if type_a != type_b:
                        interest_score += 1
                    if proj_a and proj_b and proj_a != proj_b:
                        interest_score += 2
            except:
                pass

            # Emotion check
            try:
                emotion_check = self.conn.execute(
                    'SELECT MAX(emotion) FROM nodes WHERE id IN (?, ?)',
                    (end_a, end_b)
                ).fetchone()
                if emotion_check and emotion_check[0]:
                    max_emo = emotion_check[0]
                    if max_emo > 0.5:
                        interest_score += 2
                    elif max_emo > 0.3:
                        interest_score += 1
            except:
                pass

            # Structural distance
            interest_score += min(2, (len(walk_a) + len(walk_b)) // 4)

            should_surface = interest_score >= 4

            # High-scoring dreams spawn a thought
            dream_thought = None
            if interest_score >= 6 and not any(d.get('_hasThought') for d in dreams):
                try:
                    dream_thought = self._spawn_thought(
                        f'Dream connection: "{node_end_a[:50] if node_end_a else "?"}" and "{node_end_b[:50] if node_end_b else "?"}" — found via random walks from different graph regions. Score {interest_score}. Worth investigating.',
                        [end_a, end_b] if end_a and end_b else [],
                        'dream_observation'
                    )
                except:
                    pass

            dreams.append({
                'seeds': [node_a, node_b],
                'endpoints': [node_end_a, node_end_b],
                'insight': insight,
                'intuition_id': intuition_id,
                'walk_lengths': [len(walk_a), len(walk_b)],
                'interest_score': interest_score,
                'surface': should_surface,
                'thought_id': dream_thought['id'] if dream_thought else None,
                '_hasThought': bool(dream_thought)
            })

        # Dream bridge proposals
        bridge_proposals = 0
        max_dream_proposals = 3
        try:
            recent_dreams = self.conn.execute('''
                SELECT walk_path FROM dream_log WHERE session_id = ? ORDER BY created_at DESC LIMIT ?
            ''', (session_id, DREAM_COUNT)).fetchall()
            for (walk_json,) in recent_dreams:
                if bridge_proposals >= max_dream_proposals:
                    break
                try:
                    walks = json.loads(walk_json)
                    if len(walks) >= 2:
                        walk_a, walk_b = walks[0], walks[1]
                        end_a = walk_a[-1] if walk_a else None
                        end_b = walk_b[-1] if walk_b else None
                        if end_a and end_b and end_a != end_b:
                            shared_context = 'Dream walk convergence: random walks from different regions met here'
                            proposal = self._propose_bridge(end_a, end_b, shared_context, session_id)
                            if proposal:
                                bridge_proposals += 1
                except:
                    pass
        except:
            pass

        # Surfaced insights
        surfaced_insights = [
            {
                'insight': d['insight'],
                'score': d['interest_score'],
                'endpoints': d['endpoints'],
                'intuition_id': d['intuition_id']
            }
            for d in dreams if d['surface']
        ]

        return {
            'dreams': dreams,
            'count': len(dreams),
            'bridge_proposals': bridge_proposals,
            'surfaced_insights': surfaced_insights
        }

    def _random_walk(self, start_id: str, steps: int) -> List[str]:
        """
        Weighted random walk along edges.
        Avoids loops (don't revisit nodes).
        Returns list of node IDs in path.
        """
        path = [start_id]
        current = start_id

        for _ in range(steps):
            neighbors = self.conn.execute('''
                SELECT target_id, weight FROM edges WHERE source_id = ? ORDER BY RANDOM() LIMIT 10
            ''', (current,)).fetchall()

            if not neighbors:
                break

            # Weighted random selection
            total_weight = sum(w for _, w in neighbors)
            if total_weight <= 0:
                break

            roll = random.random() * total_weight
            next_id = neighbors[0][0]
            for nid, w in neighbors:
                roll -= w
                if roll <= 0:
                    next_id = nid
                    break

            # Avoid loops
            if next_id not in path:
                path.append(next_id)
                current = next_id

        return path

    def _get_node_title(self, node_id: str) -> str:
        """Get title of a node by ID."""
        try:
            row = self.conn.execute('SELECT title FROM nodes WHERE id = ?', (node_id,)).fetchone()
            return row[0] if row else node_id
        except:
            return node_id

    def _spawn_thought(self, content: str, trigger_ids: Optional[List[str]] = None,
                      reason: str = 'prompted_by') -> Dict[str, Any]:
        """
        Create a thought node and connect it to triggers.
        Emotion=0.15 (low emotional charge).
        """
        if trigger_ids is None:
            trigger_ids = []

        # Create the thought
        title = content[:117] + '...' if len(content) > 120 else content
        result = self.remember(
            type='thought',
            title=title,
            content=content,
            keywords=f'thought brain-observation {self._extract_keywords(content)}',
            locked=False,
            emotion=0.15,
            emotion_label='curiosity',
            emotion_source='brain'
        )

        # Connect to triggers
        for trigger_id in trigger_ids:
            try:
                trigger_title = self._get_node_title(trigger_id) or trigger_id
                self.connect_typed(
                    result['id'], trigger_id, reason, 0.3, reason,
                    f'Brain noticed: "{title[:60]}" while processing "{trigger_title[:60]}"'
                )
            except:
                pass

        return result

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidation: boost stability, form bridges, mature bridge proposals, backfill embeddings.
        Returns dict with consolidated count, bridges_created, bridges_matured.
        """
        ts = self.now()
        stats = {'consolidated': 0}

        # Boost stability for nodes accessed 3+ times in last 24h
        candidates = self.conn.execute('''
            SELECT node_id, COUNT(*) as cnt FROM access_log
            WHERE timestamp > datetime(?, '-24 hours')
            GROUP BY node_id HAVING cnt >= 3
        ''', (ts,)).fetchall()

        for node_id, _ in candidates:
            self.conn.execute('''
                UPDATE nodes SET stability = stability * ?, activation = MIN(1.0, activation + 0.1),
                       updated_at = ? WHERE id = ? AND locked = 0
            ''', (STABILITY_BOOST, ts, node_id))
            stats['consolidated'] += 1

        # Promote well-connected nodes
        well_connected = self.conn.execute('''
            SELECT source_id, SUM(weight) as total_weight, COUNT(*) as edge_count
            FROM edges WHERE weight > 0.3
            GROUP BY source_id HAVING edge_count >= 5
        ''').fetchall()

        for node_id, _, _ in well_connected:
            self.conn.execute('''
                UPDATE nodes SET stability = MAX(stability, 3.0), updated_at = ? WHERE id = ? AND locked = 0
            ''', (ts, node_id))

        # Emergent bridging at consolidation (wider scan)
        try:
            bridges = self._bridge_at_consolidation()
            stats['bridges_created'] = len(bridges)
        except:
            stats['bridges_created'] = 0

        # Mature pending bridge proposals
        try:
            matured = self._mature_bridge_proposals()
            stats['bridges_matured'] = matured
        except:
            stats['bridges_matured'] = 0

        # Fire-and-forget backfill embeddings (don't wait for async)
        try:
            # Non-blocking call to backfill
            import asyncio
            asyncio.create_task(self.backfill_embeddings(20))
            stats['embeddings_backfilled'] = 'queued'
        except:
            stats['embeddings_backfilled'] = 0

        return stats

    def suggest(self, context: Optional[str] = None, file: Optional[str] = None,
               screen: Optional[str] = None, action: Optional[str] = None,
               project: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Multi-query recall with type boosts, locked node boost, file-term relevance.
        Returns dict with suggestions list and query_count.
        """
        if limit is None:
            limit = self.get_config('suggestion_limit', 5)

        queries = []

        if context:
            queries.append(context)
        if file:
            # Clean filename
            clean_file = file.replace('/', ' ').replace('\\', ' ').replace(os.path.splitext(file)[1], '')
            clean_file = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_file)
            clean_file = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean_file)
            queries.append(clean_file)
            # File parts
            file_parts = [p for p in re.split(r'[\s\-_]+', clean_file) if len(p) > 2]
            if len(file_parts) > 1:
                queries.extend(file_parts)

        if screen:
            queries.append(screen)
        if action:
            queries.append(action)

        if not queries:
            return {'suggestions': [], 'reason': 'no context provided'}

        # Run recall for each query
        seen = set()
        all_results = []
        pool_multiplier = self.get_config('recall_pool_multiplier', 2)
        recall_limit = max(limit * pool_multiplier, 15)

        for q in queries:
            result = self.recall(query=q, limit=recall_limit)
            results = result.get('results', result) if isinstance(result, dict) else result
            for r in results:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    all_results.append(r)

        # Second pass: check edge neighbors of top results for locked nodes
        try:
            top_ids = [r['id'] for r in all_results[:10]]
            if top_ids:
                placeholders = ','.join('?' * len(top_ids))
                neighbor_rows = self.conn.execute(f'''
                    SELECT DISTINCT n.id, n.type, n.title, n.content, n.keywords, n.activation,
                           n.stability, n.access_count, n.locked, n.archived, n.last_accessed, n.created_at
                    FROM edges e
                    JOIN nodes n ON (n.id = CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END)
                    WHERE (e.source_id IN ({placeholders}) OR e.target_id IN ({placeholders}))
                      AND n.locked = 1 AND n.archived = 0
                      AND n.id NOT IN ({placeholders})
                    LIMIT 20
                ''', top_ids + top_ids + top_ids).fetchall()

                for row in neighbor_rows:
                    nid = row[0]
                    if nid not in seen:
                        seen.add(nid)
                        all_results.append({
                            'id': row[0], 'type': row[1], 'title': row[2], 'content': row[3],
                            'keywords': row[4], 'activation': row[5], 'stability': row[6],
                            'access_count': row[7], 'locked': row[8] == 1, 'archived': row[9] == 1,
                            'last_accessed': row[10], 'created_at': row[11],
                            '_edge_neighbor': True
                        })
        except:
            pass

        # Project filter
        if project:
            all_results.sort(key=lambda a: (
                -(1 if a.get('project') == project else 0),
                -(a.get('effective_activation') or 0)
            ))

        # Scoring
        rule_boost = self.get_config('boost_rule', 1.3)
        decision_boost = self.get_config('boost_decision', 1.2)
        locked_boost = self.get_config('boost_locked', 1.5)
        edge_neighbor_penalty = self.get_config('penalty_edge_neighbor', 0.85)
        file_relevance_max = self.get_config('file_relevance_bonus', 0.15)

        # File-specific terms
        file_terms = set()
        if file:
            clean = file.replace('/', ' ').replace('\\', ' ').replace(os.path.splitext(file)[1], '')
            clean = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean)
            clean = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean)
            for t in clean.lower().split(r'[\s\-_]+'):
                if len(t) > 2:
                    file_terms.add(t)
        if screen:
            file_terms.add(screen.lower())

        ranked = []
        for r in all_results:
            boost = 1.0
            if r.get('type') == 'rule':
                boost = rule_boost
            if r.get('type') == 'decision':
                boost = decision_boost
            if r.get('locked'):
                boost *= locked_boost
            if r.get('_edge_neighbor'):
                boost *= edge_neighbor_penalty

            # File relevance
            file_relevance = 0
            if file_terms:
                node_text = f"{r.get('title', '')} {r.get('keywords', '')}".lower()
                for term in file_terms:
                    if term in node_text:
                        file_relevance += 1
                file_relevance = (file_relevance / len(file_terms)) * file_relevance_max

            r['suggest_score'] = ((r.get('effective_activation') or 0.5) + file_relevance) * boost
            ranked.append(r)

        ranked.sort(key=lambda r: -r['suggest_score'])

        # Locked node promotion
        selected = ranked[:limit]
        selected_ids = {r['id'] for r in selected}

        if file_terms:
            missed_locked = [
                r for r in ranked
                if r['id'] not in selected_ids and r.get('locked') and
                   r.get('type') in ('rule', 'decision') and
                   any(t in f"{r.get('title', '')} {r.get('keywords', '')}".lower() for t in file_terms)
            ]

            for locked_node in missed_locked:
                worst_idx = -1
                worst_score = float('inf')
                for i in range(len(selected) - 1, -1, -1):
                    if not selected[i].get('locked') and selected[i]['suggest_score'] < worst_score:
                        worst_idx = i
                        worst_score = selected[i]['suggest_score']
                if worst_idx >= 0:
                    selected[worst_idx] = locked_node
                    selected_ids.add(locked_node['id'])
                else:
                    break

            selected.sort(key=lambda r: -r['suggest_score'])

        suggestions = [
            {
                'id': r['id'],
                'type': r.get('type'),
                'title': r.get('title'),
                'content': r.get('content', '')[:300] if r.get('content') else None,
                'locked': r.get('locked', False),
                'relevance': r['suggest_score'],
                'reason': self._suggest_reason(r, queries)
            }
            for r in selected
        ]

        # Log suggestion
        try:
            self.conn.execute('''
                INSERT INTO suggest_log (session_id, context, suggested_ids, created_at)
                VALUES (?, ?, ?, ?)
            ''', ('auto', ' | '.join(queries), json.dumps([s['id'] for s in suggestions]), ts))
        except:
            pass

        return {'suggestions': suggestions, 'query_count': len(queries)}

    def _suggest_reason(self, node: Dict[str, Any], queries: List[str]) -> str:
        """Generate reason string for suggestion."""
        lower_title = (node.get('title') or '').lower()
        for q in queries:
            terms = [w for w in q.lower().split() if len(w) > 2]
            for t in terms:
                if t in lower_title:
                    return f'matches "{t}" from context'
        return 'related via graph connections'

    def context_boot(self, user: str, project: str, task: Optional[str] = None,
                     hints: Optional[str] = None) -> Dict[str, Any]:
        """
        3-tier progressive loading for context boot.
        Full content for top locked nodes, title-only index for rest,
        recent nodes, task-recalled nodes.
        Returns dict with brain_version, locked, recalled, recent, reset_count, last_session_note.
        """
        max_locked = CONTEXT_BOOT_LOCKED_LIMIT
        max_recall = CONTEXT_BOOT_RECALL_LIMIT
        max_recent = CONTEXT_BOOT_RECENT_LIMIT

        query_parts = [user, project, task, hints]
        query = ' '.join(p for p in query_parts if p)

        # 1. Get locked nodes with full content for top N
        locked = self.conn.execute('''
            SELECT id, type, title, content, keywords FROM nodes
            WHERE locked = 1 AND archived = 0
            ORDER BY
              CASE type WHEN 'rule' THEN 0 WHEN 'decision' THEN 1 ELSE 2 END,
              access_count DESC, last_accessed DESC
            LIMIT ?
        ''', (max_locked,)).fetchall()

        results = {
            'locked': [],
            'locked_index': [],
            'recalled': [],
            'recent': []
        }

        seen = set()

        for r in locked:
            seen.add(r[0])
            results['locked'].append({
                'id': r[0], 'type': r[1], 'title': r[2],
                'content': r[3], 'keywords': r[4]
            })

        # Title-only index for remaining locked nodes
        locked_index = self.conn.execute('''
            SELECT id, type, title FROM nodes
            WHERE locked = 1 AND archived = 0
            ORDER BY
              CASE type WHEN 'rule' THEN 0 WHEN 'decision' THEN 1 ELSE 2 END,
              access_count DESC, last_accessed DESC
            LIMIT 500 OFFSET ?
        ''', (max_locked,)).fetchall()

        for r in locked_index:
            if r[0] not in seen:
                seen.add(r[0])
                results['locked_index'].append({
                    'id': r[0], 'type': r[1], 'title': r[2]
                })

        # 2. Recently accessed nodes
        recent = self.conn.execute('''
            SELECT id, type, title, content, keywords, activation, last_accessed FROM nodes
            WHERE archived = 0 AND locked = 0
            ORDER BY last_accessed DESC LIMIT ?
        ''', (max_recent,)).fetchall()

        for r in recent:
            if r[0] not in seen:
                seen.add(r[0])
                results['recent'].append({
                    'id': r[0], 'type': r[1], 'title': r[2],
                    'content': r[3]
                })

        # 3. Recall by context query
        if query:
            recall_result = self.recall(query=query, limit=max_recall)
            recalled = recall_result.get('results', recall_result) if isinstance(recall_result, dict) else recall_result
            for r in recalled:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    results['recalled'].append({
                        'id': r['id'], 'type': r.get('type'),
                        'title': r.get('title'), 'content': r.get('content')
                    })

        # Get total locked count
        total_locked = self.conn.execute(
            'SELECT COUNT(*) FROM nodes WHERE locked = 1 AND archived = 0'
        ).fetchone()[0]

        # Find last session note
        session_logs = self.conn.execute('''
            SELECT id, title, content, created_at FROM nodes
            WHERE title LIKE '%Session%Log%Reset%' AND archived = 0
            ORDER BY created_at DESC LIMIT 1
        ''').fetchone()

        last_session_note = None
        reset_count = 0
        if session_logs:
            last_session_note = {
                'id': session_logs[0],
                'title': session_logs[1],
                'content': session_logs[2],
                'created_at': session_logs[3]
            }
            # Parse reset number
            match = re.search(r'Reset #(\d+)', session_logs[1])
            reset_count = int(match.group(1)) if match else 0

        return {
            'brain_version': BRAIN_VERSION,
            'total_nodes': self._get_node_count(),
            'total_edges': self._get_edge_count(),
            'total_locked': total_locked,
            'locked_shown': len(results['locked']),
            'has_more_locked': total_locked > max_locked,
            'reset_count': reset_count,
            'last_session_note': last_session_note,
            **results
        }

    def health_check(self, session_id: str = 'boot', auto_fix: bool = True) -> Dict[str, Any]:
        """
        Check brain health: unresolved compaction boundaries, high miss rate,
        orphaned locked nodes, stale contexts, stale staged learnings.
        Auto-fix: enrich missed nodes, promote staged learnings.
        """
        issues = []
        actions = []
        ts = self.now()

        # 1. Check for unresolved compaction boundary warnings
        boundaries = self.conn.execute('''
            SELECT id, title, created_at FROM nodes
            WHERE type = 'context' AND title LIKE '%Compaction boundary%' AND archived = 0
        ''').fetchall()

        for row in boundaries:
            issues.append({
                'type': 'compaction_boundary',
                'severity': 'high',
                'message': f'Unresolved compaction boundary from {row[2]}. Recap encoding may have been skipped.',
                'node_id': row[0]
            })

        # 2. Check for recent miss logs
        miss_count_row = self.conn.execute(
            "SELECT COUNT(*) FROM miss_log WHERE created_at > datetime('now', '-24 hours')"
        ).fetchone()
        miss_count = miss_count_row[0] if miss_count_row else 0

        if miss_count > 3:
            issues.append({
                'type': 'high_miss_rate',
                'severity': 'medium',
                'message': f'{miss_count} recall misses in the last 24 hours. Consider keyword enrichment.'
            })

        # 3. Check for orphaned locked nodes
        orphaned = self.conn.execute('''
            SELECT n.id, n.title FROM nodes n
            WHERE n.locked = 1 AND n.archived = 0
            AND n.id NOT IN (SELECT source_id FROM edges)
            AND n.id NOT IN (SELECT target_id FROM edges)
            LIMIT 5
        ''').fetchall()

        if orphaned:
            issues.append({
                'type': 'orphaned_locked_nodes',
                'severity': 'low',
                'message': f'{len(orphaned)} locked nodes with no connections.',
                'nodes': [{'id': r[0], 'title': r[1]} for r in orphaned]
            })

        # 4. Check for stale context nodes
        stale_count_row = self.conn.execute('''
            SELECT COUNT(*) FROM nodes
            WHERE type = 'context' AND locked = 0 AND archived = 0
            AND created_at < datetime('now', '-7 days')
        ''').fetchone()
        stale_count = stale_count_row[0] if stale_count_row else 0

        if stale_count > 10:
            issues.append({
                'type': 'stale_contexts',
                'severity': 'low',
                'message': f'{stale_count} context nodes older than 7 days.'
            })
            if auto_fix:
                self.conn.execute('''
                    UPDATE nodes SET archived = 1
                    WHERE type = 'context' AND locked = 0 AND archived = 0
                    AND created_at < datetime('now', '-14 days')
                ''')
                actions.append('Auto-archived context nodes older than 14 days')

        # 5. Auto-enrich keywords on missed nodes
        if auto_fix:
            try:
                missed_nodes = self.conn.execute('''
                    SELECT DISTINCT expected_node_id FROM miss_log
                    WHERE expected_node_id IS NOT NULL
                    ORDER BY rowid DESC LIMIT 10
                ''').fetchall()
                enriched = 0
                for (node_id,) in missed_nodes:
                    try:
                        self.enrich_keywords(node_id)
                        enriched += 1
                    except:
                        pass
                if enriched > 0:
                    actions.append(f'Auto-enriched keywords on {enriched} frequently-missed nodes')
            except:
                pass

        # 6. Auto-promote staged learnings
        if auto_fix:
            try:
                promoted = self.auto_promote_staged(revisit_threshold=3)
                if promoted.get('promoted', 0) > 0:
                    actions.append(f'Auto-promoted {promoted["promoted"]} staged learnings (3+ revisits)')
            except:
                pass

        # 7. Check for stale pending staged learnings
        try:
            stale_staged_row = self.conn.execute('''
                SELECT COUNT(*) FROM staged_learnings
                WHERE status = 'pending' AND created_at < datetime('now', '-7 days')
            ''').fetchone()
            stale_staged_count = stale_staged_row[0] if stale_staged_row else 0
            if stale_staged_count > 0:
                issues.append({
                    'type': 'stale_staged_learnings',
                    'severity': 'medium',
                    'message': f'{stale_staged_count} staged learnings unreviewed for 7+ days.'
                })
        except:
            pass

        # Log health check
        try:
            self.conn.execute('''
                INSERT INTO health_log (session_id, check_type, result, actions_taken, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, 'boot_check', json.dumps(issues), json.dumps(actions), ts))
        except:
            pass

        return {
            'healthy': not any(i['severity'] == 'high' for i in issues),
            'issues': issues,
            'actions': actions,
            'checked_at': ts
        }

    def _find_bridge_candidates(self, node_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find bridge candidates: 2-hop shared neighbor analysis.
        Returns nodes that share >= threshold neighbors but no direct edge.
        """
        threshold = self.get_config('bridge_threshold', 2)
        max_per_node = self.get_config('bridge_max_per_node', 5)

        # Check existing bridge count
        existing = self.conn.execute('''
            SELECT COUNT(*) FROM edges
            WHERE (source_id = ? OR target_id = ?) AND edge_type = 'emergent_bridge'
        ''', (node_id, node_id)).fetchone()
        current_bridge_count = existing[0] if existing else 0

        if current_bridge_count >= max_per_node:
            return []

        slots_left = max_per_node - current_bridge_count

        # Find 2-hop neighbors
        candidates = self.conn.execute(f'''
            SELECT second_hop.id, COUNT(DISTINCT mid.id) as shared_count,
                   second_hop.title, second_hop.type,
                   GROUP_CONCAT(mid.title, ' | ') as shared_titles
            FROM (
              SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as id
              FROM edges e
              WHERE (e.source_id = ? OR e.target_id = ?) AND e.weight >= 0.1
            ) AS neighbor
            JOIN nodes mid ON mid.id = neighbor.id AND mid.archived = 0
            JOIN edges e2 ON (e2.source_id = neighbor.id OR e2.target_id = neighbor.id) AND e2.weight >= 0.1
            JOIN nodes second_hop ON second_hop.id = CASE WHEN e2.source_id = neighbor.id THEN e2.target_id ELSE e2.source_id END
              AND second_hop.id != ?
              AND second_hop.archived = 0
            WHERE second_hop.id NOT IN (
              SELECT CASE WHEN e3.source_id = ? THEN e3.target_id ELSE e3.source_id END
              FROM edges e3
              WHERE e3.source_id = ? OR e3.target_id = ?
            )
            GROUP BY second_hop.id
            HAVING shared_count >= ?
            ORDER BY shared_count DESC
            LIMIT ?
        ''', (node_id, node_id, node_id, node_id, node_id, node_id, node_id, threshold, min(limit, slots_left))).fetchall()

        return [
            {
                'targetId': r[0],
                'sharedCount': r[1],
                'targetTitle': r[2],
                'targetType': r[3],
                'sharedTitles': r[4] or ''
            }
            for r in candidates
        ]

    def _create_bridge(self, source_id: str, target_id: str, shared_titles: str = '') -> Optional[Dict[str, Any]]:
        """
        Create a bridge edge between source and target.
        Returns created edge info or None if bridge already exists.
        """
        # Check no direct edge already exists
        existing = self.conn.execute(
            'SELECT weight FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        ).fetchone()

        if existing:
            return None

        # Get titles
        src_title = self._get_node_title(source_id) or source_id
        tgt_title = self._get_node_title(target_id) or target_id

        # Auto-generate description
        shared_part = f' via shared neighbors: {shared_titles[:150]}' if shared_titles else ''
        description = f'Emergent bridge: "{src_title[:60]}" ↔ "{tgt_title[:60]}"{shared_part}'

        self.connect_typed(source_id, target_id, 'emergent_bridge', 0.15, 'emergent_bridge', description)

        return {'sourceId': source_id, 'targetId': target_id, 'description': description, 'weight': 0.15}

    def _bridge_at_consolidation(self) -> List[Dict[str, Any]]:
        """
        Broader bridging sweep during consolidation.
        Scans recent + random nodes, creates bridges, spawns thoughts for high-shared-count pairs.
        """
        max_bridges = self.get_config('bridge_max_per_consolidation', 5)
        scan_size = 20
        created = []
        max_thoughts = 2

        # Recency-biased node selection
        recent = self.conn.execute('''
            SELECT id FROM nodes WHERE archived = 0 AND locked = 0
            AND last_accessed > datetime('now', '-48 hours')
            ORDER BY RANDOM() LIMIT ?
        ''', (max(1, scan_size // 2),)).fetchall()

        random_nodes = self.conn.execute('''
            SELECT id FROM nodes WHERE archived = 0 AND locked = 0
            ORDER BY RANDOM() LIMIT ?
        ''', (scan_size,)).fetchall()

        # Merge with dedup
        seen_ids = set()
        merged = []
        for source in [recent, random_nodes]:
            for (node_id,) in source:
                if node_id not in seen_ids and len(merged) < scan_size:
                    seen_ids.add(node_id)
                    merged.append(node_id)

        thoughts = 0
        for node_id in merged:
            if len(created) >= max_bridges:
                break
            candidates = self._find_bridge_candidates(node_id, limit=2)
            for c in candidates:
                if len(created) >= max_bridges:
                    break
                bridge = self._create_bridge(node_id, c['targetId'], c.get('sharedTitles', ''))
                if bridge:
                    created.append(bridge)
                    # Spawn thought for high shared-count
                    if c['sharedCount'] >= 4 and thoughts < max_thoughts:
                        try:
                            src_title = self._get_node_title(node_id) or node_id
                            tgt_title = c['targetTitle'] or c['targetId']
                            self._spawn_thought(
                                f'Cluster forming: "{src_title}" and "{tgt_title}" share {c["sharedCount"]} neighbors ({c.get("sharedTitles", "")[:100]}). These areas are converging.',
                                [node_id, c['targetId']],
                                'cluster_observation'
                            )
                            thoughts += 1
                        except:
                            pass

        return created

    def _propose_bridge(self, source_id: str, target_id: str, shared_titles: str = '',
                       dream_session_id: str = '') -> Optional[Dict[str, Any]]:
        """
        Propose a bridge into maturation queue (used by dream).
        Bridge won't be created until maturation timer expires.
        """
        max_pending = self.get_config('bridge_dream_max_pending', 10)

        # Check current pending count
        try:
            count_row = self.conn.execute(
                "SELECT COUNT(*) FROM bridge_proposals WHERE status = 'pending'"
            ).fetchone()
            current = count_row[0] if count_row else 0
            if current >= max_pending:
                return None
        except:
            return None

        # Check no direct edge exists
        existing = self.conn.execute(
            'SELECT weight FROM edges WHERE source_id = ? AND target_id = ?',
            (source_id, target_id)
        ).fetchone()
        if existing:
            return None

        # Check not already proposed
        try:
            dup = self.conn.execute(
                "SELECT id FROM bridge_proposals WHERE source_id = ? AND target_id = ? AND status = 'pending'",
                (source_id, target_id)
            ).fetchone()
            if dup:
                return None
        except:
            return None

        maturation_minutes = self.get_config('bridge_dream_maturation_minutes', 120)
        ts = self.now()

        try:
            self.conn.execute('''
                INSERT INTO bridge_proposals
                  (source_id, target_id, shared_context, dream_session_id, status, proposed_at, matures_at)
                VALUES (?, ?, ?, ?, 'pending', ?, datetime(?, '+' || ? || ' minutes'))
            ''', (source_id, target_id, shared_titles[:300], dream_session_id, ts, ts, maturation_minutes))
        except:
            return None

        return {'sourceId': source_id, 'targetId': target_id, 'maturationMinutes': maturation_minutes}

    def _mature_bridge_proposals(self) -> int:
        """
        Promote pending bridge proposals that have matured.
        Returns count of bridges created.
        """
        matured = 0
        try:
            ready = self.conn.execute('''
                SELECT id, source_id, target_id, shared_context FROM bridge_proposals
                WHERE status = 'pending' AND matures_at <= datetime('now')
            ''').fetchall()

            for row_id, src, tgt, ctx in ready:
                bridge = self._create_bridge(src, tgt, ctx or '')
                if bridge:
                    self.conn.execute(
                        "UPDATE bridge_proposals SET status = 'created' WHERE id = ?",
                        (row_id,)
                    )
                    matured += 1
                else:
                    self.conn.execute(
                        "UPDATE bridge_proposals SET status = 'expired' WHERE id = ?",
                        (row_id,)
                    )
        except:
            pass

        return matured

    def list_staged(self, status: str = 'pending', limit: int = 20) -> Dict[str, Any]:
        """
        List staged learnings with optional status filter.
        Returns dict with staged list.
        """
        query = '''
            SELECT sl.*, n.title, n.content, n.type, n.confidence as node_confidence
            FROM staged_learnings sl
            JOIN nodes n ON sl.node_id = n.id
            WHERE n.archived = 0
        '''
        params = []

        if status != 'all':
            query += ' AND sl.status = ?'
            params.append(status)

        query += ' ORDER BY sl.created_at DESC LIMIT ?'
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return {'staged': [dict(zip(['node_id', 'status', 'times_revisited', 'title', 'content', 'type', 'confidence'], r)) for r in rows]}

    def confirm_staged(self, node_id: str, lock: bool = False, new_title: Optional[str] = None) -> Dict[str, Any]:
        """
        Confirm a staged learning — promotes it to full node.
        Bumps confidence to 0.8, removes [staged] prefix, optionally locks.
        """
        exists = self.conn.execute('SELECT id FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if not exists:
            return {'action': 'error', 'error': f'Node {node_id} not found'}

        ts = self.now()

        # Update node
        if new_title:
            self.conn.execute(
                f'UPDATE nodes SET confidence = 0.8, locked = {1 if lock else 0}, title = ?, updated_at = ? WHERE id = ?',
                (new_title, ts, node_id)
            )
        else:
            self.conn.execute(
                f'UPDATE nodes SET confidence = 0.8, locked = {1 if lock else 0}, title = REPLACE(title, "[staged] ", ""), updated_at = ? WHERE id = ?',
                (ts, node_id)
            )

        # Update staged_learnings
        self.conn.execute(
            "UPDATE staged_learnings SET status = 'confirmed', updated_at = ?, reviewed_session = ? WHERE node_id = ?",
            (ts, 'current', node_id)
        )

        return {'action': 'confirmed', 'node_id': node_id, 'confidence': 0.8, 'locked': lock}

    def dismiss_staged(self, node_id: str, reason: str = '') -> Dict[str, Any]:
        """
        Dismiss a staged learning — archives the node.
        """
        exists = self.conn.execute('SELECT id FROM nodes WHERE id = ?', (node_id,)).fetchone()
        if not exists:
            return {'action': 'error', 'error': f'Node {node_id} not found'}

        ts = self.now()
        self.conn.execute('UPDATE nodes SET archived = 1, updated_at = ? WHERE id = ?', (ts, node_id))
        self.conn.execute(
            "UPDATE staged_learnings SET status = 'dismissed', updated_at = ?, reviewed_session = ? WHERE node_id = ?",
            (ts, reason or 'current', node_id)
        )

        return {'action': 'dismissed', 'node_id': node_id}

    def auto_promote_staged(self, revisit_threshold: int = 3) -> Dict[str, Any]:
        """
        Auto-promote staged learnings with enough revisits.
        Threshold: 3+ revisits = auto-promote to confidence 0.7.
        """
        ts = self.now()
        candidates = self.conn.execute('''
            SELECT sl.node_id, sl.times_revisited, n.title
            FROM staged_learnings sl JOIN nodes n ON sl.node_id = n.id
            WHERE sl.status = 'pending' AND sl.times_revisited >= ? AND n.archived = 0
        ''', (revisit_threshold,)).fetchall()

        if not candidates:
            return {'promoted': 0}

        count = 0
        for node_id, _, _ in candidates:
            self.conn.execute(
                'UPDATE nodes SET confidence = 0.7, title = REPLACE(title, "[staged] ", ""), updated_at = ? WHERE id = ?',
                (ts, node_id)
            )
            self.conn.execute(
                "UPDATE staged_learnings SET status = 'promoted', updated_at = ? WHERE node_id = ?",
                (ts, node_id)
            )
            count += 1

        return {'promoted': count, 'threshold': revisit_threshold}

    def set_config(self, key: str, value: Any) -> Dict[str, Any]:
        """
        Set a config value in brain_meta. Persists across restarts.
        """
        ts = self.now()
        self.conn.execute(
            'INSERT OR REPLACE INTO brain_meta (key, value, updated_at) VALUES (?, ?, ?)',
            (key, str(value), ts)
        )
        return {'key': key, 'value': value, 'updated_at': ts}

    def get_suggest_metrics(self, period_days: int = 7) -> Dict[str, Any]:
        """
        Aggregate suggest_log stats for a period.
        Returns metrics on suggestion quality and diversity.
        """
        since = datetime.utcnow().timestamp() - (period_days * 86400)
        try:
            rows = self.conn.execute('''
                SELECT COUNT(*) as calls, AVG(LENGTH(suggested_ids)) as avg_pool_size
                FROM suggest_log
                WHERE created_at > datetime(?, 'unixepoch')
            ''', (since,)).fetchone()

            return {
                'period_days': period_days,
                'total_suggest_calls': rows[0] if rows else 0,
                'avg_suggestions_per_call': rows[1] if rows and rows[1] else 0
            }
        except:
            return {'period_days': period_days, 'error': 'Could not aggregate metrics'}

    def get_debug_status(self) -> Dict[str, Any]:
        """
        Check if debug mode is enabled.
        Reads from brain_meta, falls back to env var.
        """
        try:
            row = self.conn.execute("SELECT value FROM brain_meta WHERE key = 'debug_enabled'").fetchone()
            if row:
                return {'debug_enabled': row[0] == '1'}
        except:
            pass

        debug_env = os.environ.get('TMEMORY_DEBUG') == '1'
        return {'debug_enabled': debug_env}

    def log_debug(self, event_type: str, source: str, **kwargs) -> Dict[str, Any]:
        """
        Log a debug event.
        Writes to debug_log table with event metadata.
        """
        try:
            ts = self.now()
            self.conn.execute('''
                INSERT INTO debug_log
                  (session_id, event_type, source, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', ('unknown', event_type, source, json.dumps(kwargs), ts))
            return {'logged': True}
        except Exception as e:
            return {'logged': False, 'error': str(e)}

    def enrich_keywords(self, node_id: str) -> Optional[str]:
        """
        Enrich keywords on a node from its content.
        Used by health check for frequently-missed nodes.
        """
        try:
            row = self.conn.execute(
                'SELECT content, keywords FROM nodes WHERE id = ?',
                (node_id,)
            ).fetchone()
            if not row or not row[0]:
                return None

            content, existing_kw = row
            new_kw = self._extract_keywords(content)
            combined = f'{existing_kw} {new_kw}' if existing_kw else new_kw

            self.conn.execute(
                'UPDATE nodes SET keywords = ?, updated_at = ? WHERE id = ?',
                (combined, self.now(), node_id)
            )
            return combined
        except:
            return None

    def get_config(self, key: str, default_val: Any = None) -> Any:
        """
        Get a config value from brain_meta.

        Args:
            key: Config key
            default_val: Default if not found

        Returns:
            Config value or default
        """
        try:
            cursor = self.conn.execute('SELECT value FROM brain_meta WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                val = row[0]
                # Auto-parse numbers
                if default_val is not None and isinstance(default_val, (int, float)):
                    try:
                        return float(val) if '.' in str(val) else int(val)
                    except (ValueError, TypeError):
                        pass
                if val == 'true':
                    return True
                if val == 'false':
                    return False
                return val
        except Exception:
            pass
        return default_val

    # ─── Pre-Edit Batch Method ───
    # Replaces the /pre-edit HTTP endpoint from index.js
    # Combines suggest + procedures + encoding health into one call

    def pre_edit(self, file: str, tool_name: str = 'Edit') -> dict:
        """
        Batch pre-edit call combining all lookups into one.
        Replaces 8 sequential HTTP calls from the old architecture.

        Args:
            file: Filename being edited
            tool_name: 'Edit' or 'Write'

        Returns:
            Dict with suggestions, procedures, context_files, encoding health, timings
        """
        import time as _time
        t0 = _time.time()
        timings = {}

        # 1. Suggest
        t1 = _time.time()
        suggest_result = self.suggest(
            context=f"editing {file}",
            file=file,
            action=tool_name.lower(),
            limit=10
        )
        timings['suggest_ms'] = round((_time.time() - t1) * 1000)

        # 2. Procedures
        t2 = _time.time()
        proc_result = self.procedure_trigger('pre_edit', {'file': file, 'tool': tool_name})
        timings['procedures_ms'] = round((_time.time() - t2) * 1000)

        # 3. Encoding health
        activity = self._get_session_activity()
        boot_time = activity.get('boot_time', self.now())
        remembers = int(activity.get('remember_count', 0))
        edits_checked = int(activity.get('edit_check_count', 0))
        last_remember = activity.get('last_remember_at', None)

        # Compute session minutes
        try:
            from datetime import datetime as _dt
            boot_dt = _dt.fromisoformat(boot_time.replace('Z', '+00:00'))
            now_dt = _dt.now(boot_dt.tzinfo) if boot_dt.tzinfo else _dt.utcnow()
            session_minutes = (now_dt - boot_dt).total_seconds() / 60
        except Exception:
            session_minutes = 0

        # Compute minutes since last remember
        mins_since_remember = 0
        if last_remember:
            try:
                last_dt = _dt.fromisoformat(last_remember.replace('Z', '+00:00'))
                mins_since_remember = (now_dt - last_dt).total_seconds() / 60
            except Exception:
                pass

        # Determine encoding health status
        edits_since = edits_checked  # approximate — reset on each remember
        if remembers == 0 and session_minutes > 3:
            encoding_health = 'NONE'
        elif edits_since > 8 and mins_since_remember > 5:
            encoding_health = 'STALE'
        else:
            encoding_health = 'OK'

        # Increment edit check counter
        self.record_edit_check()

        # 4. Context files (nodes of type 'file' matching the edited filename)
        context_files = []
        try:
            cursor = self.conn.execute(
                "SELECT id, title, content, keywords, updated_at FROM nodes WHERE type = 'file' AND archived = 0 AND (title LIKE ? OR keywords LIKE ?) LIMIT 3",
                (f'%{file}%', f'%{file}%')
            )
            for row in cursor.fetchall():
                context_files.append({
                    'id': row[0], 'title': row[1], 'summary': (row[2] or '')[:200],
                    'topic': row[3] or '', 'last_updated': row[4],
                })
        except Exception:
            pass

        timings['total_ms'] = round((_time.time() - t0) * 1000)

        return {
            'suggestions': suggest_result.get('suggestions', []),
            'procedures': proc_result.get('matched', []),
            'context_files': context_files,
            'encoding': {
                'health': encoding_health,
                'remembers': remembers,
                'edits_since_last_remember': edits_since,
                'minutes_since_last_remember': round(mins_since_remember),
                'session_minutes': round(session_minutes),
            },
            'embedder_ready': embedder.is_ready(),
            'debug_enabled': self.get_debug_status(),
            'timings': timings,
        }

    def procedure_trigger(self, trigger_type: str, context: dict = None) -> dict:
        """
        Find and return procedures matching a trigger type.

        Args:
            trigger_type: 'session_start', 'pre_edit', 'pre_compact', etc.
            context: Optional context dict with trigger-specific data

        Returns:
            Dict with matched procedures list
        """
        context = context or {}
        matched = []

        try:
            cursor = self.conn.execute(
                "SELECT id, title, content, keywords FROM nodes WHERE type = 'procedure' AND archived = 0 AND locked = 1"
            )
            for row in cursor.fetchall():
                node_id, title, content, keywords = row
                content_lower = (content or '').lower()
                keywords_lower = (keywords or '').lower()

                # Check if procedure matches trigger type
                if trigger_type in content_lower or trigger_type in keywords_lower:
                    # Parse procedure content for steps
                    steps = content or ''
                    category = 'general'
                    if 'session_start' in keywords_lower:
                        category = 'session_start'
                    elif 'pre_edit' in keywords_lower:
                        category = 'pre_edit'
                    elif 'pre_compact' in keywords_lower:
                        category = 'pre_compact'

                    # Check file-specific procedures
                    if trigger_type == 'pre_edit' and 'file' in context:
                        file_name = context['file'].lower()
                        if file_name not in content_lower and file_name not in keywords_lower:
                            # Check for wildcard patterns
                            if '*' not in content_lower:
                                continue

                    matched.append({
                        'id': node_id,
                        'title': title,
                        'steps': steps,
                        'category': category,
                    })
        except Exception:
            pass

        return {'matched': matched}

    # ═══════════════════════════════════════════════════════════════
    # ABSORB — Merge knowledge from another brain
    # ═══════════════════════════════════════════════════════════════

    def absorb(self, source_brain: 'Brain',
               auto_merge_locked: bool = True,
               auto_merge_unlocked: bool = False,
               fuzzy_threshold: int = 3,
               dry_run: bool = False) -> Dict[str, Any]:
        """
        Absorb knowledge from another brain into this one.

        Finds nodes in source_brain that don't exist here, and merges them in.
        Designed for the Hindsight cycle: run relearning simulation → absorb
        new discoveries back into the active brain.

        Strategy:
          1. For each source node, fuzzy-match against existing titles
          2. Exact/fuzzy match → skip (already known) or flag (possible update)
          3. No match + locked → auto-absorb (high-confidence new knowledge)
          4. No match + unlocked → skip by default (context/task decay anyway)
          5. Transfer connections where both endpoints exist in target brain

        Args:
            source_brain: Brain to absorb knowledge from
            auto_merge_locked: Auto-absorb new locked nodes (default True)
            auto_merge_unlocked: Auto-absorb new unlocked nodes (default False)
            fuzzy_threshold: Min word overlap for fuzzy title matching (default 3)
            dry_run: If True, report what would happen without making changes

        Returns:
            Dict with absorbed, skipped, flagged, connections_created counts
            and detailed lists of each category
        """
        report = {
            'absorbed': [],
            'skipped': [],
            'flagged': [],
            'connections_created': 0,
            'summary': {},
        }

        # Build title index of existing nodes for fast matching
        existing = {}
        cursor = self.conn.execute('SELECT id, title, type, locked FROM nodes WHERE title IS NOT NULL')
        for row in cursor.fetchall():
            nid, title, ntype, locked = row
            existing[title] = {
                'id': nid,
                'type': ntype,
                'locked': bool(locked),
                'words': set(title.lower().split()),
            }

        # Map source node IDs → new node IDs (for connection transfer)
        id_map = {}

        # Get all source nodes
        source_nodes = source_brain.conn.execute(
            '''SELECT id, type, title, content, keywords, locked, emotion,
                      emotion_label, project, access_count
               FROM nodes WHERE title IS NOT NULL
               ORDER BY locked DESC, access_count DESC'''
        ).fetchall()

        for src_row in source_nodes:
            (src_id, src_type, src_title, src_content, src_keywords,
             src_locked, src_emotion, src_emotion_label, src_project,
             src_access) = src_row

            if not src_title:
                continue

            # ── Step 1: Match against existing nodes ──
            match_type, match_node = self._match_existing(
                src_title, existing, fuzzy_threshold
            )

            if match_type == 'exact':
                # Already have this exact title — skip
                report['skipped'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'reason': 'exact_duplicate',
                    'existing_id': match_node['id'],
                })
                id_map[src_id] = match_node['id']
                continue

            if match_type == 'fuzzy':
                # Close match — flag for review (might be update or duplicate)
                report['flagged'].append({
                    'source_title': src_title[:80],
                    'source_type': src_type,
                    'source_locked': bool(src_locked),
                    'existing_title': match_node.get('_matched_title', '')[:80],
                    'existing_id': match_node['id'],
                    'reason': 'fuzzy_match',
                })
                id_map[src_id] = match_node['id']
                continue

            # ── Step 2: New knowledge — decide whether to absorb ──
            should_absorb = False
            if src_locked and auto_merge_locked:
                should_absorb = True
            elif not src_locked and auto_merge_unlocked:
                should_absorb = True

            if not should_absorb:
                report['skipped'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'locked': bool(src_locked),
                    'reason': 'policy_skip' if not src_locked else 'auto_merge_disabled',
                })
                continue

            # ── Step 3: Absorb the node ──
            if dry_run:
                report['absorbed'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'locked': bool(src_locked),
                    'emotion': src_emotion or 0,
                    'dry_run': True,
                })
                continue

            try:
                result = self.remember(
                    type=src_type,
                    title=src_title,
                    content=src_content or '',
                    keywords=src_keywords or '',
                    locked=bool(src_locked),
                    emotion=float(src_emotion or 0),
                    emotion_label=src_emotion_label or 'neutral',
                    project=src_project or '',
                )
                new_id = result.get('id') if isinstance(result, dict) else result
                id_map[src_id] = new_id

                report['absorbed'].append({
                    'title': src_title[:80],
                    'type': src_type,
                    'locked': bool(src_locked),
                    'new_id': new_id,
                    'emotion': src_emotion or 0,
                })

                # Register in existing index for subsequent matching
                existing[src_title] = {
                    'id': new_id,
                    'type': src_type,
                    'locked': bool(src_locked),
                    'words': set(src_title.lower().split()),
                }

            except Exception as e:
                report['flagged'].append({
                    'source_title': src_title[:80],
                    'source_type': src_type,
                    'reason': f'remember_error: {e}',
                })

        # ── Step 4: Transfer connections ──
        if not dry_run:
            report['connections_created'] = self._absorb_connections(
                source_brain, id_map
            )

        # ── Summary ──
        report['summary'] = {
            'source_nodes': len(source_nodes),
            'absorbed': len(report['absorbed']),
            'skipped': len(report['skipped']),
            'flagged': len(report['flagged']),
            'connections_created': report['connections_created'],
            'dry_run': dry_run,
        }

        if not dry_run:
            self.conn.commit()

        return report

    def _match_existing(self, title: str, existing: Dict,
                        fuzzy_threshold: int = 3) -> tuple:
        """
        Match a title against existing nodes.

        Returns:
            ('exact', node_dict) — exact title match
            ('fuzzy', node_dict) — fuzzy word overlap match
            ('none', None) — no match found
        """
        # Exact match
        if title in existing:
            return ('exact', existing[title])

        # Fuzzy match: check word overlap
        title_words = set(title.lower().split())
        best_overlap = 0
        best_match = None
        best_title = ''

        for ex_title, ex_node in existing.items():
            overlap = len(title_words & ex_node['words'])
            if overlap > best_overlap and overlap >= fuzzy_threshold:
                best_overlap = overlap
                best_match = ex_node
                best_title = ex_title

        if best_match:
            best_match['_matched_title'] = best_title
            return ('fuzzy', best_match)

        return ('none', None)

    def _absorb_connections(self, source_brain: 'Brain',
                            id_map: Dict[str, str]) -> int:
        """
        Transfer edges from source brain where both endpoints exist in target.

        Args:
            source_brain: Brain to read edges from
            id_map: Mapping of source_id → target_id

        Returns:
            Number of connections created
        """
        created = 0
        source_edges = source_brain.conn.execute(
            '''SELECT source_id, target_id, weight, relation, edge_type
               FROM edges
               WHERE edge_type IN ('related', 'part_of', 'corrected_by',
                                   'exemplifies', 'depends_on', 'produced')'''
        ).fetchall()

        for src_source, src_target, weight, relation, edge_type in source_edges:
            # Both endpoints must exist in target brain
            target_source = id_map.get(src_source)
            target_target = id_map.get(src_target)

            if not target_source or not target_target:
                continue
            if target_source == target_target:
                continue

            # Check if edge already exists
            existing = self.conn.execute(
                'SELECT 1 FROM edges WHERE source_id = ? AND target_id = ?',
                (target_source, target_target)
            ).fetchone()

            if existing:
                continue

            try:
                self.connect_typed(
                    source_id=target_source,
                    target_id=target_target,
                    relation=relation or 'related',
                    weight=weight or 0.5,
                    edge_type=edge_type or 'related',
                )
                created += 1
            except Exception:
                pass

        return created

    def save(self, backup: bool = False):
        """
        Commit pending changes and optionally back up database.

        Args:
            backup: If True, create a backup copy
        """
        self.conn.commit()

        if backup and self.db_path:
            try:
                import shutil
                backup_path = f'{self.db_path}.backup-{datetime.utcnow().strftime("%Y%m%d-%H%M%S")}'
                shutil.copy2(self.db_path, backup_path)
            except Exception as e:
                print(f'[brain] Backup failed: {e}')

    def close(self):
        """Commit, close database connection, and remove from singleton cache."""
        self.conn.commit()
        self.conn.close()
        # Remove from singleton registry if present
        canonical = os.path.realpath(self.db_path)
        with Brain._lock:
            if Brain._instances.get(canonical) is self:
                del Brain._instances[canonical]
