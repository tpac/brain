"""
brain — Shared Constants

Constants used across multiple brain mixin modules.
Extracted to avoid circular imports (mixins can't import from brain.py).
"""

import re
from datetime import datetime, timedelta

# ═══════════════════════════════════════════════════════════════
# CONSTANTS: Decay rates by node type (hours until weight halves)
# ═══════════════════════════════════════════════════════════════

DECAY_HALF_LIFE = {
    'person': 720,      # 30 days
    'project': 720,     # 30 days
    'object': 720,      # 30 days
    'decision': float('inf'),
    'rule': float('inf'),
    'concept': 168,     # 7 days
    'task': 48,         # 2 days
    'file': 168,        # 7 days
    'context': 24,      # 1 day
    'intuition': 12,    # 12 hours
    'procedure': float('inf'),
    'thought': 168,     # 7 days — thoughts need time to connect across sessions
    'fn_reasoning': float('inf'),
    'param_influence': float('inf'),
    'code_concept': 720,
    'arch_constraint': float('inf'),
    'causal_chain': 720,
    'bug_lesson': float('inf'),
    'comment_anchor': 168,
    'tension': float('inf'),
    'hypothesis': 720,
    'pattern': 1440,
    'catalyst': float('inf'),
    'aspiration': 2160,
    'performance': 720,
    'failure_mode': float('inf'),
    'capability': 720,
    'interaction': 720,
    'meta_learning': float('inf'),
    'correction': float('inf'),
    'validation': 720,
    'mental_model': 720,
    'reasoning_trace': 1440,
    'uncertainty': 168,
    'purpose': float('inf'),
    'mechanism': 720,
    'impact': float('inf'),
    'constraint': float('inf'),
    'convention': 1440,
    'lesson': float('inf'),
    'vocabulary': float('inf'),
    'boot': float('inf'),         # Boot nodes persist forever — they ARE the handoff
}

# ═══════════════════════════════════════════════════════════════
# CONFIDENCE: Type defaults and dynamics
# ═══════════════════════════════════════════════════════════════

# Default confidence by node type — how reliable each type tends to be
TYPE_CONFIDENCE = {
    'rule': 0.85, 'decision': 0.80, 'lesson': 0.85, 'correction': 0.95,
    'constraint': 0.85, 'convention': 0.80, 'procedure': 0.85,
    'purpose': 0.80, 'mechanism': 0.75, 'impact': 0.75,
    'mental_model': 0.65, 'hypothesis': 0.50, 'uncertainty': 0.40,
    'concept': 0.70, 'context': 0.60, 'task': 0.70,
    'intuition': 0.40, 'thought': 0.35,
    'pattern': 0.60, 'tension': 0.55, 'aspiration': 0.50,
    'person': 0.85, 'project': 0.80, 'file': 0.75,
    'vocabulary': 0.90, 'validation': 0.90, 'boot': 0.90,
    # Legacy code cognition types
    'fn_reasoning': 0.75, 'param_influence': 0.70, 'code_concept': 0.70,
    'arch_constraint': 0.85, 'causal_chain': 0.70, 'bug_lesson': 0.85,
    'comment_anchor': 0.80,
    'capability': 0.70, 'reasoning_trace': 0.65,
}

# Keywords that suggest a node is about external systems (faster confidence decay)
EXTERNAL_CLAIM_KEYWORDS = {
    'api', 'sdk', 'version', 'v1', 'v2', 'v3', 'v4', 'v5',
    'library', 'package', 'framework', 'tool', 'plugin',
    'supports', 'doesnt support', 'cannot', "can't", 'not possible',
    'limitation', 'workaround', 'deprecat', 'breaking change',
    'release', 'update', 'upgrade', 'migration',
    'claude code', 'openai', 'github', 'npm', 'pip', 'docker',
}

# Stability floor
STABILITY_FLOOR_ACCESS_THRESHOLD = 5
STABILITY_FLOOR_RETENTION = 0.3

# Hebbian learning
LEARNING_RATE = 0.2
MAX_WEIGHT = 1.0
PRUNE_THRESHOLD = 0.05

# Recency scoring weights
RECENCY_WEIGHT = 0.30
RELEVANCE_WEIGHT = 0.35
FREQUENCY_WEIGHT = 0.10
EMOTION_WEIGHT = 0.25

# Emotion constants
EMOTION_FLOOR = 0.3
EMOTION_DECAY_RATE = 0.95

# Recency time bands
RECENCY_BANDS = [
    {'maxHours': 1,     'score': 1.0},
    {'maxHours': 6,     'score': 0.9},
    {'maxHours': 24,    'score': 0.75},
    {'maxHours': 72,    'score': 0.5},
    {'maxHours': 168,   'score': 0.3},
    {'maxHours': 720,   'score': 0.15},
    {'maxHours': float('inf'), 'score': 0.05},
]

# Page sizes
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
CONTEXT_BOOT_LOCKED_LIMIT = 50
CONTEXT_BOOT_RECALL_LIMIT = 15
CONTEXT_BOOT_RECENT_LIMIT = 10

# Embeddings weights
EMBEDDING_PRIMARY_WEIGHT = 0.90
KEYWORD_FALLBACK_WEIGHT = 0.10
TFIDF_SEMANTIC_WEIGHT = EMBEDDING_PRIMARY_WEIGHT
TFIDF_KEYWORD_WEIGHT = KEYWORD_FALLBACK_WEIGHT

TFIDF_STOP_WORDS = {
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of',
    'that', 'this', 'it', 'be', 'as', 'by', 'from', 'has', 'was', 'are', 'were', 'been', 'have', 'had',
    'not', 'they', 'their', 'will', 'would', 'can', 'could', 'do', 'does', 'did', 'should', 'may',
    'might', 'shall', 'than', 'into', 'about', 'also', 'its', 'just', 'more', 'other', 'some', 'such',
    'then', 'there', 'these', 'what', 'when', 'where', 'who', 'how', 'all', 'each', 'both', 'few',
    'most', 'any', 'own', 'same', 'over', 'only', 'very', 'after', 'before', 'between', 'under',
    'above', 'out', 'up', 'down', 'use', 'used', 'using', 'new', 'like', 'get', 'set', 'one', 'two',
}

# Encoding version — bumps when encoding pipeline changes.
# Stored on each node so recall knows encoding quality.
CURRENT_ENCODING_VERSION = "v5"

# Relevance floor — version-aware. Enriched nodes get higher bar.
# Sweep data (214 enriched cases): 0.80 → 87/89 pos, 32/54 neg blocked.
# Bare nodes (no enrichments) use lower bar to stay visible.
RELEVANCE_FLOOR_ENRICHED = 0.80   # top result won via enrichment vector
RELEVANCE_FLOOR_PRIMARY = 0.50    # top result won via primary embedding only

# Vocabulary expansion
VOCAB_EXPANSION_MAX = 3  # Max terms added per query via vocabulary expansion
VOCAB_GENERIC_THRESHOLD = 0.05  # Reject vocab terms matching >5% of nodes

# Extended stop words — includes common verbs that aren't domain-specific
EXTENDED_STOP_WORDS = TFIDF_STOP_WORDS | {
    'working', 'make', 'run', 'build', 'check', 'create', 'start', 'stop',
    'need', 'want', 'thing', 'change', 'move', 'find', 'help', 'keep',
    'try', 'show', 'call', 'put', 'think', 'look', 'take', 'give', 'say',
    'come', 'go', 'see', 'know', 'good', 'bad', 'big', 'small', 'part',
    'work', 'way', 'time', 'done', 'made', 'feature',
}

# Intent detection patterns
INTENT_PATTERNS = {
    'decision_lookup': re.compile(r'\b(what did (?:we|tom|i) (?:decide|choose|pick)|decision about|decided on)\b', re.IGNORECASE),
    'reasoning_chain': re.compile(r'\b(why did (?:we|i)|reason for|reasoning behind|what led to|how come)\b', re.IGNORECASE),
    'state_query': re.compile(r"\b(what(?:'s| is) the (?:current|latest)|status of|state of|where (?:are|is) (?:we|it))\b", re.IGNORECASE),
    'temporal': re.compile(r'\b(when did|last (?:week|month|time|session)|this (?:week|month)|before (?:the|we)|after (?:the|we)|yesterday|today|recently|history of|timeline)\b', re.IGNORECASE),
    'correction_lookup': re.compile(r'\b(what mistake|lesson(?:s)? learned|correction|what went wrong|what did (?:we|i) learn|mistakes?\b.*learn|learn(?:ed)? from)\b', re.IGNORECASE),
    'how_to': re.compile(r"\b(how (?:do|does|to|should)|what(?:'s| is) the (?:best|right) way)\b", re.IGNORECASE),
    'list_query': re.compile(r'\b(list (?:all|every)|show me (?:all|every)|what are (?:all|the))\b', re.IGNORECASE),
}

INTENT_TYPE_BOOSTS = {
    'decision_lookup':   {'decision': 1.5, 'rule': 1.0, 'lesson': 1.2, 'correction': 1.3},
    'reasoning_chain':   {'decision': 1.3, 'rule': 1.2, 'context': 1.1, 'mechanism': 1.3, 'reasoning_trace': 1.4, 'mental_model': 1.2},
    'state_query':       {'context': 1.5, 'project': 1.3, 'task': 1.3, 'object': 1.4, 'purpose': 1.2},
    'temporal':          {'decision': 1.0, 'context': 1.2},
    'correction_lookup': {'decision': 1.5, 'rule': 1.2, 'correction': 1.5, 'lesson': 1.3},
    'how_to':            {'rule': 1.5, 'decision': 1.2, 'mechanism': 1.5, 'convention': 1.4, 'purpose': 1.2, 'constraint': 1.3},
    'list_query':        {'rule': 1.0, 'decision': 1.0, 'object': 1.3},
    'general':           {'purpose': 1.1, 'mechanism': 1.1, 'impact': 1.1, 'vocabulary': 1.1},
}

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
            'after': (datetime.utcnow() - timedelta(days=3)).isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\blast session\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': (datetime.utcnow() - timedelta(hours=6)).isoformat() + 'Z'
        }
    },
]

# Edge types
EDGE_TYPES = {
    'reasoning_step': {'defaultWeight': 0.9, 'decays': False, 'description': 'Step N to Step N+1'},
    'produced': {'defaultWeight': 0.85, 'decays': False, 'description': 'Reasoning chain to Decision'},
    'corrected_by': {'defaultWeight': 0.85, 'decays': False, 'description': 'Correction event to corrector'},
    'exemplifies': {'defaultWeight': 0.8, 'decays': True, 'halfLife': 720, 'description': 'Decision to Rule'},
    'part_of': {'defaultWeight': 0.7, 'decays': False, 'description': 'Node to parent'},
    'depends_on': {'defaultWeight': 0.7, 'decays': False, 'description': 'Node requires another'},
    'related': {'defaultWeight': 0.5, 'decays': False, 'description': 'Manual or inferred — intentional, no decay'},
    'co_accessed': {'defaultWeight': 0.3, 'decays': True, 'halfLife': 336, 'description': 'Hebbian co-recall — 14d half-life (v2: up from 7d)'},
    'emergent_bridge': {'defaultWeight': 0.15, 'decays': True, 'halfLife': 720, 'description': 'Auto-discovered bridge — 30d half-life (v2: up from 3d, high quality conceptual links)'},
}

# Edge decay
EDGE_PRUNE_THRESHOLD = 0.1  # Edges below this weight after decay are deleted

# Critical node safety
CRITICAL_BOOST = 3.0              # Recall score multiplier for critical=1 nodes
CRITICAL_SIMILARITY_THRESHOLD = 0.20  # Lowered embedding threshold for critical nodes

# Graph traversal
SPREAD_DECAY = 0.5
MAX_HOPS = 3
MAX_NEIGHBORS = 50
STABILITY_BOOST = 1.5

# B.2: Graph-augmented recall (1-hop typed neighbors)
GRAPH_AUGMENT_TOP_N = 5       # Augment from top N embedding results
NEIGHBOR_DAMPEN = 0.6          # Neighbors score at 60% of parent
INTENTIONAL_EDGE_TYPES = {
    'related', 'about', 'part_of', 'depends_on', 'implements', 'contains',
    'enables', 'constrains', 'governs', 'extends', 'describes', 'corrected_by',
    'produced', 'addresses', 'elaborates', 'informed_by', 'exemplifies',
    'evolved', 'questions', 'traced', 'tests', 'foundation_for',
}
GRAPH_NEIGHBOR_LIMIT = 10      # Max neighbors per parent node

# V5 Multi-vector enrichment (Embedding Migration to LLM)
# At encode time, an LLM generates enrichment vectors for each node:
# Q (question), A (anchor), B (bridge), K (keywords).
# These are embedded and searched alongside the primary embedding.
# Benchmark: NDCG 0.701, 93/104 passed (vs baseline 0.204, 34/104).
ENRICHMENT_NEIGHBOR_COUNT = 5    # Number of neighbors to include in enrichment prompt
ENRICHMENT_VECTOR_TYPES = ('question', 'anchor', 'bridge', 'keywords')

# V5 Structured prompt template (V2 variant — proven best for small LLMs)
# Small models need constrained structured prompts, NOT motivational framing.
# See: "Lesson: small LLMs ignore context, cant follow motivational framing"
ENRICHMENT_PROMPT_TEMPLATE = """The brain found these related memories:
{neighbors}

New node: "{title}"
Content: "{content}"

Generate exactly these lines, no explanations:
Q: [one question a user would naturally ask that leads to this node]
A: [3-5 word phrase using words from the neighbors above]
B: [one sentence connecting this node to its most important neighbor]
K: [5 comma-separated keywords borrowed from neighbors that also describe this node]"""

# Dreaming
DREAM_WALK_LENGTH = 5
DREAM_COUNT = 3
DREAM_MIN_NOVELTY = 2

# Evolution / Curiosity
REASONING_STEP_TYPES = ['observation', 'hypothesis', 'attempt', 'evidence', 'failure', 'feedback', 'decision', 'lesson']
CURIOSITY_MAX_PROMPTS = 3
CURIOSITY_CHAIN_GAP_THRESHOLD = 0
CURIOSITY_DECAY_WARNING_HOURS = 18
