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
# Z-SCORE: Contrastive recall scoring
# Measures SURPRISE (how unusual this cosine is for this node)
# rather than raw similarity. Hub nodes with high mean get flattened.
# ═══════════════════════════════════════════════════════════════

ZSCORE_ENABLED = True
ZSCORE_MIN_STD = 0.01               # Floor to avoid division by zero
ZSCORE_STATS_KEY_MEAN = 'zscore_mean'   # node_metadata_kv key for stored mean
ZSCORE_STATS_KEY_STD = 'zscore_std'     # node_metadata_kv key for stored std
ZSCORE_DEFAULT_MEAN = 0.50           # Fallback for nodes without precomputed stats
ZSCORE_DEFAULT_STD = 0.05            # Fallback std (conservative — minimal normalization)

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

# Hebbian learning
LEARNING_RATE = 0.2
MAX_WEIGHT = 1.0
PRUNE_THRESHOLD = 0.05

# ═══════════════════════════════════════════════════════════════
# UNIFIED SCORING — recall_scoring.py
# ═══════════════════════════════════════════════════════════════
# Semantic similarity is the gatekeeper (multiplicative).
# Other signals MODULATE within bounded ranges.
# Zero semantic = zero final, regardless of other signals.
#
# Biology mapping:
#   semantic_score → pattern completion (hippocampal CA3)
#   freshness      → hippocampal fresh trace (created_at, NOT last_accessed)
#   emotion        → amygdala modulation (GANE: winner-take-more)
#   frequency      → cue overload penalty (high-access = low diagnostic)
#   confidence     → consolidation strength
#
# Formula: final = base * (1.0 + recency + emotion + frequency + confidence)
# Max modulator: 1.545x. Min modulator: 0.81x. Bounded.
#
# Tested 2026-04-02: fixes R@8 gap where embedding improvements (+20pts
# in simulation) showed +0pts in production due to additive scoring.

# Freshness from creation time — when the knowledge was born.
# NOT last_accessed (self-fulfilling: mark_accessed refreshes every recall cycle).
FRESHNESS_BANDS = [
    {'max_hours': 1,              'boost': 0.30},  # just created
    {'max_hours': 6,              'boost': 0.25},  # same session
    {'max_hours': 24,             'boost': 0.20},  # today
    {'max_hours': 72,             'boost': 0.12},  # this week
    {'max_hours': 168,            'boost': 0.06},  # recent
    {'max_hours': 720,            'boost': 0.02},  # this month
    {'max_hours': float('inf'),   'boost': 0.0},   # established knowledge
]

# Emotion amplification — multiplicative on semantic base.
# abs(emotion) * this = boost. Max 0.2 boost at emotion=1.0.
EMOTION_AMPLIFICATION = 0.20

# Frequency PENALTY — high access_count = hub = low diagnostic value.
# Kicks in above threshold. Capped at max.
FREQUENCY_PENALTY_THRESHOLD = 20   # Below this: no penalty
FREQUENCY_PENALTY_SCALE = 0.03     # log(ac/threshold) * this
FREQUENCY_PENALTY_MAX = 0.10       # Cap: never more than 10% penalty

# Confidence modulator — validated knowledge gets mild boost.
# Maps [0.1, 1.0] confidence to [-0.09, +0.045] boost.
CONFIDENCE_NEUTRAL = 0.70          # No effect at default confidence
CONFIDENCE_SCALE = 0.15            # (confidence - neutral) * this


# Page sizes
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
CONTEXT_BOOT_LOCKED_LIMIT = 50
CONTEXT_BOOT_RECALL_LIMIT = 15
CONTEXT_BOOT_RECENT_LIMIT = 10

# Boot context rendering
BOOT_COMMUNITY_TOP = 18            # Top communities by size/maturity
BOOT_COMMUNITY_RECENT = 2          # Recently worked-on communities
BOOT_IDENTITY_LIMIT = 3            # Identity nodes in YOU section
BOOT_IDENTITY_CONTENT_LIMIT = 300  # Content chars per identity node

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

# Relevance floor — per-result minimum score.
# v8.7: Lowered from 0.80/0.50 and changed from all-or-nothing to per-result.
# Previous sweep data (214 enriched cases) was pre-enrichment-cap, no longer applies.
# The Haiku distiller is the primary quality gate; this floor catches obvious noise.
# Relevance floors — minimum blended score to include in results.
# v8.7: Per-result floor (not global). Each result must meet its own bar.
RELEVANCE_FLOOR_ENRICHED = 0.45   # result won via enrichment vector
RELEVANCE_FLOOR_PRIMARY = 0.25    # result won via primary embedding only

# Enrichment cap — enrichments boost primary score, don't replace it.
# v8.7: Enrichment vectors (question, anchor, bridge, keywords) were overriding
# primary content similarity, causing broad nodes to win every query.
# Cap means: enrichment can add at most 30% of gap above primary.
# Example: primary=0.45, enrichment=0.82 → 0.45 + 0.3*(0.82-0.45) = 0.561
ENRICHMENT_CAP = 0.30

# Title-match boost — proportional to fraction of query terms found in node title.
# When query terms appear in a node's title, it's a strong signal of relevance.
# Boost is additive: score += title_fraction * TITLE_MATCH_BOOST
TITLE_MATCH_BOOST = 0.3

# Noise floor — minimum blended score for non-critical candidates.
# v9: Raised from 0.05 to cut pure embedding noise before scoring.
NOISE_FLOOR_THRESHOLD = 0.15

# FTS5 full-text search
FTS5_CANDIDATE_LIMIT = 5     # Max FTS5-only candidates sent to surfacer
FTS5_SEARCH_LIMIT = 30       # How many FTS5 hits to fetch before filtering
FTS5_PASSTHROUGH_SCORE = 0.20  # Score for FTS5-only candidates (above noise floor)

# Retrieval stats — surfacer guidance thresholds
RETRIEVAL_LOW_CONFIDENCE = 0.35   # Top score below this → surfacer gets low-confidence warning

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

def _start_of_today():
    """Midnight UTC today."""
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

def _start_of_week():
    """Monday 00:00 UTC of the current week."""
    now = datetime.utcnow()
    return (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

def _start_of_month():
    """First day of the current month, 00:00 UTC."""
    return datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

TEMPORAL_PATTERNS = [
    {
        'pattern': re.compile(r'\btoday\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': _start_of_today().isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\byesterday\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': (_start_of_today() - timedelta(days=1)).isoformat() + 'Z',
            'before': _start_of_today().isoformat() + 'Z',
        }
    },
    {
        'pattern': re.compile(r'\bthis week\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': _start_of_week().isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\blast week\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': (_start_of_week() - timedelta(weeks=1)).isoformat() + 'Z',
            'before': _start_of_week().isoformat() + 'Z',
        }
    },
    {
        'pattern': re.compile(r'\bthis month\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': _start_of_month().isoformat() + 'Z'
        }
    },
    {
        'pattern': re.compile(r'\blast month\b', re.IGNORECASE),
        'range_fn': lambda: {
            'after': (_start_of_month().replace(day=1) - timedelta(days=1)).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'before': _start_of_month().isoformat() + 'Z',
        }
    },
    {
        'pattern': re.compile(r'\blast (\d+) days?\b', re.IGNORECASE),
        'range_fn': lambda m: {
            'after': (datetime.utcnow() - timedelta(days=int(m.group(1)))).isoformat() + 'Z'
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
    'co_accessed': {'defaultWeight': 0.3, 'decays': True, 'halfLife': 720, 'description': 'Judge-selected Hebbian — meaningful co-activation, participates in traversal'},
    'emergent_bridge': {'defaultWeight': 0.15, 'decays': True, 'halfLife': 720, 'description': 'Auto-discovered bridge — excluded from traversal'},
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
# STABILITY_BOOST, STABILITY_FLOOR_* removed 2026-04-13 — stability field deprecated.

# B.2: Graph-augmented recall — 3-degree traversal
GRAPH_AUGMENT_TOP_N = 5       # Seed traversal from top N embedding results

# Edge type sets
INTENTIONAL_EDGE_TYPES = {
    'related', 'about', 'part_of', 'depends_on', 'implements', 'contains',
    'enables', 'constrains', 'governs', 'extends', 'describes', 'corrected_by',
    'produced', 'addresses', 'elaborates', 'informed_by', 'exemplifies',
    'evolved', 'questions', 'traced', 'tests', 'foundation_for',
    'related_to', 'caused_by', 'supports', 'blocks', 'example_of', 'evolved_from',
    'contradicts', 'refers_to',
}
# Edges excluded from graph traversal during recall.
# co_accessed: NOW surface-selected only (clean). Old noise edges deleted 2026-04-02.
# emergent_bridge: Auto-discovered weak connections — excluded from traversal.
EXCLUDED_EDGE_TYPES = {'emergent_bridge'}

# Situation embeddings — WHEN knowledge matters (second vector dimension)
SITUATION_WEIGHT = 0.2          # Additive boost for situation match during recall
SITUATION_THRESHOLD = 0.30      # Min cosine similarity to count as situation match

# 3-degree traversal settings
TRAVERSE_DEPTH = 3
TRAVERSE_DAMPEN = [0.6, 0.3, 0.12]       # Score dampening per degree
TRAVERSE_LIMITS = [8, 6, 4]               # Max neighbors fetched per degree
TRAVERSE_SEMANTIC_BONUS = 0.15            # Additive bonus when graph + embedding converge
TRAVERSE_SEMANTIC_THRESHOLD = 0.30        # Min cosine sim to earn semantic bonus
TRAVERSE_CONVERGENCE_BOOST = 0.3          # Per additional parent: score *= 1.0 + N * this

# Temporal freshness multipliers for graph-discovered nodes
FRESHNESS_MULTIPLIERS = {
    'today': 1.2,       # revised/created in last 24h
    'this_week': 1.0,   # last 7 days
    'this_month': 0.8,  # last 30 days
    'older': 0.6,       # 30+ days
}

# Backward compat alias
NEIGHBOR_DAMPEN = TRAVERSE_DAMPEN[0]
GRAPH_NEIGHBOR_LIMIT = TRAVERSE_LIMITS[0]

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

# Dream, Evolution, Curiosity constants removed 2026-04-13 — all systems deleted.


# ═══════════════════════════════════════════════════════════════
# S2 MAINTENANCE SCHEDULING — brain.run_maintenance_if_due()
# ═══════════════════════════════════════════════════════════════
# When does the brain consider running S2? Two conditions:
#
# 1. Daemon has been idle (no requests) for MAINTENANCE_IDLE_THRESHOLD_SECONDS.
#    Pausing during active work is the signal the operator isn't typing —
#    safer to spend API budget now than mid-turn.
# 2. MAINTENANCE_MIN_INTERVAL_SECONDS have passed since the previous run,
#    tracked in brain_meta (s2_last_run_ts). Persists across daemon
#    restarts so reboots don't re-trigger maintenance immediately.
#
# In steady state — no new encoding → no new merge/place/heal candidates —
# a fire is cheap: decoders scan, find 0 work, encoders skip. So tune these
# for "how often to catch backlog during active use," not "how cheap is
# a no-op fire."
MAINTENANCE_IDLE_THRESHOLD_SECONDS = 3 * 60   # 3 min idle → more chances
                                              # to fire during active work
MAINTENANCE_MIN_INTERVAL_SECONDS = 15 * 60    # 15 min between runs
                                              # (steady state caps ~96/day)
