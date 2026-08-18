"""
brain — Shared Constants

Constants used across multiple brain mixin modules.
Extracted to avoid circular imports (mixins can't import from brain.py).
"""

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

# Page sizes
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
CONTEXT_BOOT_LOCKED_LIMIT = 50
CONTEXT_BOOT_RECALL_LIMIT = 15
CONTEXT_BOOT_RECENT_LIMIT = 10

# 2026-05-02 (Frame Phase 2.5): BOOT_COMMUNITY_TOP, BOOT_COMMUNITY_RECENT,
# BOOT_IDENTITY_LIMIT, BOOT_IDENTITY_CONTENT_LIMIT removed — they sized
# the old recall-driven YOU/OPERATOR/BRAIN MAP boot sections that the
# Frame-centered render_boot_v2 replaced. No callers left.

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

# Stopwords for the BRAIN_TITLE_BOOST='idf2' arm. df-over-titles misprices
# conversational words (titles are noun-phrases: 'does' df=58 scores as rare as
# 'fatigue' df=41) — these never earn a title boost regardless of idf.
_TITLE_BOOST_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at',
    'for', 'with', 'from', 'by', 'as', 'over', 'under', 'about', 'into',
    'out', 'up', 'down', 'off', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'am', 'do', 'does', 'did', 'done', 'have', 'has', 'had',
    'will', 'would', 'can', 'could', 'should', 'shall', 'may', 'might',
    'how', 'what', 'whats', 'when', 'where', 'which', 'who', 'whom', 'why',
    'that', 'this', 'these', 'those', 'there', 'here', 'i', 'me', 'my',
    'we', 'our', 'us', 'you', 'your', 'yours', 'it', 'its', 'they', 'them',
    'their', 'he', 'she', 'his', 'her', 'again', 'also', 'just', 'so',
    'too', 'not', 'no', 'yes', 'if', 'then', 'than', 'any', 'all', 'some',
    'more', 'most', 'other', 'very',
})

# Noise floor — minimum blended score for non-critical candidates.
# v9: Raised from 0.05 to cut pure embedding noise before scoring.
NOISE_FLOOR_THRESHOLD = 0.15

# FTS5 full-text search
FTS5_CANDIDATE_LIMIT = 5     # Max FTS5-only candidates sent to surfacer
FTS5_SEARCH_LIMIT = 30       # How many FTS5 hits to fetch before filtering
FTS5_PASSTHROUGH_SCORE = 0.20  # Score for FTS5-only candidates (above noise floor)

# Trace-chain lane — episodic dual-store rescue (flag-gated via BRAIN_TRACE_CHAIN=1, default OFF).
# Design: docs/RECALL-DUAL-STORE-DESIGN.md §3.2 + §3.3 form 1. Tier-1 proven (dual_store_merge_probe:
# #11 0->8 EX.CO rescued against the real buried baseline). Mirrors the fts5_only reserved-lane shape.
TRACE_CHAIN_RESERVE = 5      # reserved tail slots for trace-chain rescues (additive; never reorders top)
TRACE_CHAIN_T = 5            # top dialogue traces to chain FROM (answer trace may not be rank-1)
TRACE_CHAIN_N = 25           # nodes each trace pulls before dedup/merge

# recall_episodes — episodic pull over the traces layer (TraceDAL.filter_events +
# optional semantic re-rank over the existing trace_embeddings). trace→trace, returns
# full episode records — distinct from the trace→node TRACE_CHAIN rescue lane above.
EPISODE_DEFAULT_LIMIT = 10           # episodes returned by default
EPISODE_MAX_LIMIT = 500              # hard cap on a single pull
EPISODE_DEFAULT_WINDOW_DAYS = 7      # default created_at lower bound when no session/time scope
                                     # (perf: bounds the contains LIKE scan over a big append-only table)
EPISODE_SEMANTIC_CANDIDATE_CAP = 500  # max candidates fetched before semantic (query) re-rank
# EPISODE_RENDER_BODY_CHARS moved to trace_contract.TRACE_BODY_CHARS (2026-06-29) —
# the body cap is now a shared trace-render config, not an episode-only constant.

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

# Edge types
EDGE_TYPES = {
    'reasoning_step': {'defaultWeight': 0.9, 'decays': False, 'description': 'Step N to Step N+1'},
    'produced': {'defaultWeight': 0.85, 'decays': False, 'description': 'Reasoning chain to Decision'},
    'corrected_by': {'defaultWeight': 0.85, 'decays': False, 'description': 'Correction event to corrector'},
    'exemplifies': {'defaultWeight': 0.8, 'decays': True, 'halfLife': 720, 'description': 'Decision to Rule'},
    'part_of': {'defaultWeight': 0.7, 'decays': False, 'description': 'Node to parent'},
    'depends_on': {'defaultWeight': 0.7, 'decays': False, 'description': 'Node requires another'},
    'related': {'defaultWeight': 0.5, 'decays': False, 'description': 'Manual or inferred — intentional, no decay'},
}
# co_accessed + emergent_bridge retired 2026-08-17 (nodes ab56d25a, 072e26d8).
# Existing rows remain in the DB until the separate backed-up deletion ships;
# the exclusion lists that hide them from reads stay until then.

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
# Traversal exclusions derive from the noise aspect at load time —
# brain.aspects.structural_exclusions.

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
MAINTENANCE_MIN_INTERVAL_SECONDS = 60 * 60    # 1 hour between runs — throttled
                                              # to reduce S2-vs-S1 contention
                                              # (S2 cycles run minutes and share
                                              # the org rate limit with surface)
# In addition to the time gate, require at least this many S1 Encoder (Scribe)
# runs since the last S2 run. S2's material is encoded nodes — a recall (S1
# Surface) reads the graph and creates nothing to consolidate, and surfaces
# fire across many parallel streams. So cadence ties to ENCODING, not recalls:
# no new encoded nodes → nothing for S2 to do → don't fire. The Scribe runs
# every 5th conversational turn, so 2 runs ≈ 10 turns of new material. The
# FORCE_FIRE valve below still overrides this for multi-day stalls.
MAINTENANCE_MIN_ENCODE_RUNS = 2
# Safety valve: if maintenance hasn't fired in this long, force-fire on the
# next poll regardless of idle. Catches multi-day stalls (observed 2.5d gaps
# between 04-28 and 04-30 when the idle gate never opened).
MAINTENANCE_FORCE_FIRE_SECONDS = 24 * 60 * 60

# Boot grace period (added 2026-05-08): never fire S2 maintenance during the
# first N seconds after daemon start, regardless of idle/min-interval state.
# Why: maintenance with `last_activity_ts == 0.0` was treated as infinitely
# idle and fired on the very first daemon poll, which made first-recall-
# after-restart consistently time out (consolidation encoder held the write
# lock for tens of seconds while making LLM calls). Boot should not be
# overloaded with many things — the user gets a working brain immediately;
# maintenance can wait its turn after the first interaction.
MAINTENANCE_BOOT_GRACE_SECONDS = 90

# Log retention + orphan sweep cadence, run as a DBMaintenance task. Housekeeping
# with no deadline — hourly reclaims space long before it matters. It lived in the
# Claude Code idle hook until that event stopped firing and it went six weeks
# without running; the scheduler thread has no such dependency.
LOGS_MAINTENANCE_INTERVAL_S = 60 * 60

# ── LLM rejection backoff ──
# How long LLM features stay paused after the provider REFUSES a call (a dead
# key, an exhausted quota). A refusal costs no tokens, so this ladder trades
# resume latency against noise, not money: an operator who disables and
# re-enables the same key resumes within the hour with no restart, while a key
# that stays dead stops being hammered after the first few minutes. Escalates
# per consecutive rejection; the last value is the ceiling.
LLM_REJECT_BACKOFF_MINUTES = (5, 15, 30, 60)

# Strikes age out after a quiet stretch longer than the ladder's ceiling, so a
# rejection weeks from now starts at 5 minutes instead of inheriting an hour.
LLM_REJECT_STRIKE_RESET_SECONDS = 2 * 60 * 60

# ── LLM transport ──
# Hard upper bound on any single Anthropic SDK call (S1 surface, S1 encode, S2
# encoders, scouts). The SDK default is roughly 600s but is measured against
# time.monotonic(), which does NOT advance while the process is suspended
# (macOS sleep). A call started right before sleep can therefore hang
# indefinitely after wake. A post-sleep hang is recovered reactively
# (ensure_daemon at session start / the MCP health monitor during a session,
# both force-restarting via launchctl kickstart -k); this constant bounds
# normal-mode hangs (slow API, throttled response, etc.) so a stuck call
# doesn't tie up a worker forever. Community encoder round 2 on cold-cache
# batches can legitimately take ~218s; 600s leaves headroom without inviting
# silence.
#
# It lives here rather than in scales/runner.py because it is daemon-level
# transport policy, not encoder-lane detail: brain.py builds the shared client
# with it too, and was reaching down into the encoder lane for it.
ANTHROPIC_CLIENT_TIMEOUT = 600.0

# Recall-lane query expansion gets its OWN, much tighter bound. It is a
# best-effort ~1s Haiku call on the recall hot path, and a stall there blocks a
# recall worker thread — so the encoder lane's 600s ceiling is the wrong shape
# entirely. 15s is >10x the observed call time: generous enough that a slow but
# working call still lands, short enough that a hung socket costs one recall
# instead of ten minutes. Paired with max_retries=0 at the call site, since
# recall proceeds on the primary query when expansion misses.
RECALL_EXPANSION_TIMEOUT_S = 15.0

# ── Dashboard (keyless-onboarding notices) ──
# The dashboard's own default lives in dashboard/server.py (deliberately
# servers-decoupled); this is the DAEMON-side single source for building the
# /setup URL in keyless notices — daemon_hooks (per-turn PAUSED block) and
# brain_voice (boot banner) both read it here instead of re-inlining the
# literal port (code review 2026-07-17, contract-first).
DASHBOARD_DEFAULT_PORT = 47303


def dashboard_setup_url() -> str:
    """The /setup URL keyless notices point users at.

    Reads DASHBOARD_PORT from the environment (populated from
    ~/.config/brain/env by load_env on the keyless path, so a custom port
    is honored daemon-side) with the fixed default as fallback.
    """
    import os
    return "http://localhost:%s/setup" % (
        os.environ.get('DASHBOARD_PORT') or DASHBOARD_DEFAULT_PORT)
