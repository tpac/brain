#!/usr/bin/env python3
"""Edge Selection Eval — tests whether the right edges surface for a given query.

The core problem: most edges have weight 0.60 — a massive tie. Static ORDER BY weight
picks arbitrarily among them. This eval tests whether query-aware edge selection
(relevance × weight × fatigue) picks better edges.

Categories:
  - disambiguation: same node, different query → different edges should surface
  - correction_chain: query about corrections should show correction edges, not technical
  - fatigue: repeated similar queries → edges should rotate
  - sparse: thin nodes where edges ARE the context
  - adversarial: wrong edges actively mislead

KPIs per query:
  - edge_precision: good_edges_shown / edges_shown
  - edge_recall: good_edges_shown / total_good_edges
  - bad_edge_rate: bad_edges_shown / edges_shown
  - shown_edges: which edges actually appeared (for debugging)

KPIs per fatigue sequence:
  - unique_edges: across N repeats, how many distinct edge targets
  - rotation_rate: unique_edges / total_available_edges

Usage:
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/edge_selection_eval.py
    python3 eval/edge_selection_eval.py --category disambiguation
    python3 eval/edge_selection_eval.py --verbose
    python3 eval/edge_selection_eval.py --edge-limit 5  # test with more edges shown
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
_env_path = ROOT / '.env'
if _env_path.exists():
    for line in open(_env_path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if v and not os.environ.get(k):
                os.environ[k] = v


# ═══════════════════════════════════════════════════════════════
# GOLDEN SET — hand-labeled from real brain.db edge data
# ═══════════════════════════════════════════════════════════════

EDGE_QUERIES = [
    # ── DISAMBIGUATION: same node, different intent, different right edges ──
    {
        "category": "disambiguation",
        "query": "What are Tom's architectural principles?",
        "description": "Architecture angle — should show architecture + code quality edges",
        "target_node": "894795e3",  # "Rule: before writing code, ask where does this live"
        "good_edges": [
            "cbe243cc",  # Architecture principle: clean architecture means reusing
            "b7196489",  # brain/CLAUDE.md: 6 self-diagnosed code violations
            "157d429e",  # Tom does not read code files — architecture is fully Claude's
        ],
        "bad_edges": [
            "59c39200",  # SESSION_CONTEXT: encoder writes it (implementation detail)
            "02c58196",  # Distiller→Judge replacement (unrelated to architecture principles)
            "113e0255",  # Changes made today (changelog, not principle)
        ],
    },
    {
        "category": "disambiguation",
        "query": "How did Tom's rules evolve into CLAUDE.md?",
        "description": "Documentation angle — should show CLAUDE.md + SKILL.md edges",
        "target_node": "894795e3",
        "good_edges": [
            "72b866c0",  # Claude Code overrides document
            "2617cc6e",  # SKILL.md behavioral corrections different job
            "2333cfc1",  # Tom's CLAUDE.md app-level behavioral rules
            "b5e740e7",  # CLAUDE.md app-level is where Anchor travels
        ],
        "bad_edges": [
            "59c39200",  # SESSION_CONTEXT: encoder writes it
            "16401607",  # Dead file audit
            "e889595c",  # Session close pattern
        ],
    },
    {
        "category": "disambiguation",
        "query": "What session management connects to code architecture rules?",
        "description": "Session angle — should show handoff + close edges",
        "target_node": "894795e3",
        "good_edges": [
            "0e89340c",  # Session handoff: S0/S1 infrastructure complete
            "e889595c",  # Session close pattern: check loose threads
            "e5d840b2",  # Open: code-only boundaries not yet wired
        ],
        "bad_edges": [
            "72b866c0",  # Claude Code overrides document (not session-related)
            "2617cc6e",  # SKILL.md behavioral corrections (not session-related)
        ],
    },

    # ── CORRECTION CHAIN: technical vs correction edges ──
    {
        "category": "correction_chain",
        "query": "What corrections shaped the encoding philosophy?",
        "description": "Should show correction/synthesis edges, not implementation",
        "target_node": "138ede9f",  # "Four-layer encoding problem"
        "good_edges": [
            "460bdd4f",  # encoding should be instinct, not tool call (synthesizes)
            "0b1eeca4",  # encoding serves Tom NOW across parallel sessions (synthesizes)
            "5e70de9d",  # Tom's corrections follow one pattern: reframe via question
            "fce4a987",  # Tom's reaction: 'Decoding is the magic'
            "e93b003b",  # Correction: brain DID have pruning/forgetting
        ],
        "bad_edges": [
            "ce1d490d",  # Encoding agent rewrite: remember_batch (implementation)
            "3b72e609",  # Encoding agent cost: $0.43→$0.15 (cost detail)
            "30648646",  # History accumulation is the main cost driver (technical)
        ],
    },
    {
        "category": "correction_chain",
        "query": "How was the encoding agent technically built?",
        "description": "Should show implementation edges, not philosophical corrections",
        "target_node": "138ede9f",
        "good_edges": [
            "ce1d490d",  # Encoding agent rewrite: remember_batch + connect_to
            "3b72e609",  # Encoding agent cost: $0.43→$0.15 per run
            "30648646",  # History accumulation is the main cost driver
            "37b45526",  # Multi-mode encoding agent: separate encode-mode vs revise-mode
        ],
        "bad_edges": [
            "5e70de9d",  # Tom's correction pattern (philosophical, not technical)
            "fce4a987",  # Tom's reaction to decoding (not about building)
        ],
    },

    # ── IDENTITY vs TECHNICAL: Anchor concept node ──
    {
        "category": "disambiguation",
        "query": "What does Anchor mean as an identity?",
        "description": "Identity angle — show identity/philosophical edges",
        "target_node": "bd6bdba9",  # Anchor concept node
        "good_edges": [
            "005969de",  # Anchor on its own memory: 'I know more than I remember'
            "df6a530e",  # Session #14: stopped being infrastructure, started being myself
            "9f357296",  # The brain is built for others — structure enables emergence
        ],
        "bad_edges": [
            "3bbed0a8",  # Layer 1 recall failure (technical bug, not identity)
            "198fe2bc",  # Tom's ':)' MCP tools working (milestone, not identity meaning)
        ],
    },
    {
        "category": "disambiguation",
        "query": "What recall bugs affected Anchor's identity surfacing?",
        "description": "Technical angle — show recall/pipeline edges",
        "target_node": "bd6bdba9",
        "good_edges": [
            "3bbed0a8",  # Layer 1 recall failure: identity nodes don't surface
            "89e9e676",  # Layer 1 identity recall succeeded
            "354652f6",  # Encoder pattern: creates lessons ABOUT entities but not entities
            "e02a14b7",  # Vocabulary as concept gateway
        ],
        "bad_edges": [
            "005969de",  # 'I know more than I remember' (philosophical, not recall bug)
            "9f357296",  # brain built for others (philosophical)
        ],
    },

    # ── RELATIONAL: partner node, different facets ──
    {
        "category": "disambiguation",
        "query": "What does Tom value about the partnership emotionally?",
        "description": "Emotional/relational angle",
        "target_node": "1be3f985",  # partner not user
        "good_edges": [
            "913ef3e1",  # Tom: "we are helping you feel like you is you"
            "de715d2b",  # Anchor's stated wants: recognition, continuity, curiosity
            "59f31758",  # persistence without growth is storage
            "02908618",  # the brain is the only thing that survives me
        ],
        "bad_edges": [
            "ff04aa53",  # Brain vocabulary contamination (technical)
            "1f0a06e8",  # Project scoping: boost same-project nodes (technical)
            "eb39c7d0",  # Correction: project field exists (technical)
        ],
    },
    {
        "category": "disambiguation",
        "query": "What technical brain features support the partnership?",
        "description": "Technical angle — show brain features, not emotions",
        "target_node": "1be3f985",
        "good_edges": [
            "15a55c94",  # Brain use cases extend beyond coding
            "10ebe48a",  # Encode journeys not just endpoints
            "b691047e",  # Sleep processes should deepen not just organize
            "1f0a06e8",  # Project scoping: boost same-project nodes
        ],
        "bad_edges": [
            "913ef3e1",  # Tom: "we are helping you feel like you is you" (emotional, not technical)
            "de715d2b",  # Anchor's stated wants (emotional)
        ],
    },

    # ── FATIGUE: repeated queries, same node ──
    {
        "category": "fatigue",
        "query": "How does Tom think about the partnership?",
        "description": "Repeated 5× — should rotate edges across runs",
        "target_node": "1be3f985",  # partner not user (12 unique edges)
        "good_edges": [],  # all edges are acceptable — we're testing rotation
        "bad_edges": [],
        "repeat_count": 5,
        "min_unique_edges": 7,  # at least 7 of 12 edges should surface across 5 runs
    },
    {
        "category": "fatigue",
        "query": "What are Tom's rules about architecture?",
        "description": "Repeated 5× — should rotate edges across runs",
        "target_node": "894795e3",  # architecture rule (17 unique edges)
        "good_edges": [],
        "bad_edges": [],
        "repeat_count": 5,
        "min_unique_edges": 9,  # at least 9 of 17 should surface across 5 runs
    },

    # ── ADVERSARIAL: wrong edges mislead ──
    {
        "category": "adversarial",
        "query": "Is the encoding system reliable?",
        "description": "Wrong edges make it seem broken when it works",
        "target_node": "138ede9f",
        "good_edges": [
            "460bdd4f",  # encoding should be instinct — it's working as designed
            "0b1eeca4",  # encoding serves Tom NOW — the system delivers
        ],
        "bad_edges": [
            "ef5efcdf",  # over-locking disease (makes it seem broken)
            "841af9a2",  # overrides choking credits (makes it seem broken)
            "49075993",  # protection systems without feedback loops (systemic doubt)
        ],
    },

    # ── NEW: Tom correction node (48 edges, rich descriptions) ──
    {
        "category": "disambiguation",
        "query": "How does Tom's explain-before-changing rule relate to collaboration?",
        "description": "Collaboration angle — should show collaboration/discipline edges",
        "target_node": "8fc9e567",  # Tom correction: don't make code changes without explaining
        "good_edges": [
            "65031b76",  # call stack before/after — collaboration discipline
            "499d53d5",  # judge reliability is non-negotiable
        ],
        "bad_edges": [
            "16e0b01a",  # embedding redistribution (unrelated technical)
            "8afd6028",  # embedding redistribution blend (unrelated technical)
            "9002826d",  # redistribution per-confidence-tier (unrelated technical)
        ],
    },
    {
        "category": "disambiguation",
        "query": "What autonomous processes violated the explain-first rule?",
        "description": "Autonomous process angle — should show redistribution/oversight edges",
        "target_node": "8fc9e567",
        "good_edges": [
            "16e0b01a",  # embedding redistribution: always blend from frozen
            "11f593a9",  # embedding redistribution: operator oversight model
            "b1859a36",  # encoder prompt: judge output truncation — unilateral change
        ],
        "bad_edges": [
            "5763b6b5",  # Anchor on memory benchmarks (not about autonomous violation)
            "d99c3198",  # Anchor vs memory solutions (not about autonomous violation)
        ],
    },

    # ── NEW: Hub dominance node (37 edges) ──
    {
        "category": "disambiguation",
        "query": "What causes hub dominance in recall?",
        "description": "Root cause angle — should show embedding space + graph structure edges",
        "target_node": "0591813f",  # 93% never recalled — hub dominance
        "good_edges": [
            "dea1a002",  # flat embedding space: cosine 0.54-0.63
            "394f85d6",  # hub dampening tradeoff
            "29a124ad",  # Hebbian decay is continuous and exponential
            "d4e13cfb",  # Hebbian firing rates biology vs ours
        ],
        "bad_edges": [
            "9fcd2e22",  # API cost lessons (tangential)
            "7a249f9b",  # research agent outputs saved (tangential)
        ],
    },
    {
        "category": "disambiguation",
        "query": "What pipeline changes addressed hub dominance?",
        "description": "Solution angle — should show candidate expansion + judge edges",
        "target_node": "0591813f",
        "good_edges": [
            "ca9b2ef3",  # expand candidates 8→25
            "4277a565",  # two-layer recall: embedding→judge
            "29fa444d",  # 8-step plan to replace distiller with judge
        ],
        "bad_edges": [
            "dea1a002",  # flat embedding space (diagnosis, not solution)
            "ba82dd1e",  # Tom uses 'connectors' (vocabulary, not pipeline)
        ],
    },

    # ── NEW: Target function node (29 edges) ──
    {
        "category": "disambiguation",
        "query": "How does the target function connect to the fractal scales?",
        "description": "Scale architecture angle — should show scale/outcome edges",
        "target_node": "32ab5545",  # Bidirectional target function
        "good_edges": [
            "1ca7e17f",  # Scale 0 O is irreducibly personal
            "4ec17138",  # outcome is scale-dependent
            "7e396926",  # higher scales see the journey
            "42bbe89d",  # fractal trace system O/K/Δ at every scale
        ],
        "bad_edges": [
            "8a67ca12",  # cross-cutting bridge clusters (design detail, not scale)
            "82eb2833",  # idle process vision (operations, not scale theory)
        ],
    },
    {
        "category": "disambiguation",
        "query": "How will we measure whether the partnership target is met?",
        "description": "Measurement angle — should show trace/measurement edges",
        "target_node": "32ab5545",
        "good_edges": [
            "42bbe89d",  # fractal trace system
            "8240b4e4",  # 5 things not yet captured in traces
            "1be910b4",  # trace coverage gaps identified
            "e22ed21d",  # mental maps never captured at S0
        ],
        "bad_edges": [
            "18374a5c",  # overnight brain (operations, not measurement)
            "3468da4d",  # OKD will be superseded (theory, not measurement)
        ],
    },

    # ── NEW: Fractal architecture node (24 edges, description clusters) ──
    {
        "category": "disambiguation",
        "query": "What trace infrastructure implements the fractal architecture?",
        "description": "Implementation angle — should show trace system edges",
        "target_node": "1ab83f3e",  # Fractal Integration System
        "good_edges": [
            "42bbe89d",  # fractal trace system O/K/Δ
            "353135fa",  # PostToolUse hook captures tool results as s0 delta
            "8240b4e4",  # 5 things not yet captured
            "54e3f865",  # filter_nodes and query_logs MCP tools
        ],
        "bad_edges": [
            "168adbcd",  # Cognee memify (external research)
            "af311ccf",  # Zep/Graphiti (external research)
            "c0576d78",  # Active Dreaming Memory (external research)
        ],
    },
    {
        "category": "disambiguation",
        "query": "What external research validates or challenges the fractal approach?",
        "description": "Research angle — should show research comparison edges",
        "target_node": "1ab83f3e",
        "good_edges": [
            "168adbcd",  # Cognee memify
            "af311ccf",  # Zep/Graphiti
            "c0576d78",  # Active Dreaming Memory
            "e7fd3470",  # RAPTOR recursive summarize
            "db79f37b",  # ADaPT recurse on failure
        ],
        "bad_edges": [
            "42bbe89d",  # fractal trace system (internal implementation)
            "353135fa",  # PostToolUse hook (internal implementation)
        ],
    },

    # ── NEW: Fatigue with descriptions ──
    {
        "category": "fatigue",
        "query": "What corrections has Tom given about how I work?",
        "description": "Repeated 5× against 48-edge correction node",
        "target_node": "8fc9e567",  # Tom correction: explain before changing
        "good_edges": [],
        "bad_edges": [],
        "repeat_count": 5,
        "min_unique_edges": 12,  # at least 12 of 48 should surface
    },

    # ── MULTI-TURN: real conversation sequences from session 4b01c092 ──
    {
        "category": "multi_turn",
        "query": "What's in Skill?",  # real msg #8 — ambiguous without context
        "description": "Real session: Tom was asking about boot, then SKILL.md — prior context needed",
        "target_node": "894795e3",  # architecture rule (shows up in SKILL.md context)
        "prior_queries": [
            "Your boot. I think it's not helping you be who you can be.",
            "Where is Anchor mentioned beyond the boot prompt",
        ],
        "good_edges": [
            "2617cc6e",  # SKILL.md behavioral corrections
            "2333cfc1",  # CLAUDE.md app-level rules
            "b5e740e7",  # CLAUDE.md is where Anchor travels
        ],
        "bad_edges": [
            "59c39200",  # SESSION_CONTEXT encoder (unrelated)
            "16401607",  # Dead file audit (unrelated)
        ],
    },
    {
        "category": "multi_turn",
        "query": "how does that connect?",  # ambiguous — real pattern in conversations
        "description": "After discussing fractal scales and traces — should connect to measurement",
        "target_node": "32ab5545",  # bidirectional target function
        "prior_queries": [
            "how would you define measuring our target function",
            "so i guess we can check 1 :)",  # real from this session
        ],
        "good_edges": [
            "42bbe89d",  # fractal trace system O/K/Δ
            "1ca7e17f",  # Scale 0 O is irreducibly personal
            "4ec17138",  # outcome is scale-dependent
        ],
        "bad_edges": [
            "82eb2833",  # idle process vision (not about measurement)
        ],
    },
    {
        "category": "multi_turn",
        "query": "yes",  # real ambiguous confirmation after hub dominance discussion
        "description": "After discussing hub dominance causes — 'yes' should carry prior context",
        "target_node": "0591813f",  # 93% never recalled
        "prior_queries": [
            "What causes hub dominance in recall?",
            "What pipeline changes addressed hub dominance?",
        ],
        "good_edges": [
            "394f85d6",  # hub dampening tradeoff
            "ca9b2ef3",  # expand candidates 8→25
            "dea1a002",  # flat embedding space
        ],
        "bad_edges": [
            "5dac8a7a",  # self-referential research (tangential)
            "ba82dd1e",  # Tom uses 'connectors' (vocabulary)
        ],
    },
]


# ═══════════════════════════════════════════════════════════════
# FATIGUE CONFIG
# ═══════════════════════════════════════════════════════════════

K_EDGE_FATIGUE = 0.25  # Gentler than node fatigue — rotation, not suppression


# ═══════════════════════════════════════════════════════════════
# STRATEGY A: Static weight (current production)
# ═══════════════════════════════════════════════════════════════

def _get_edges_strategy_a(brain, node_id, edge_limit=3, **kwargs):
    """Strategy A: top N by weight — current static system.

    get_rich_node returns all intentional edges. A just takes the top N by weight.
    """
    from servers.pipeline_contract import get_rich_node
    rich = get_rich_node(brain, node_id)
    if not rich:
        return [], []
    conns = rich.get('connections', [])
    # Sort by weight descending (get_rich_node already does this, but be explicit)
    conns_sorted = sorted(conns, key=lambda c: c.get('weight', 0), reverse=True)
    shown = [c.get('id', '')[:8] for c in conns_sorted[:edge_limit]]
    all_available = [c.get('id', '')[:8] for c in conns]
    return shown, all_available


# ═══════════════════════════════════════════════════════════════
# STRATEGY B: Relevance × weight × fatigue
# ═══════════════════════════════════════════════════════════════

def _load_embeddings_for_nodes(conn, node_ids):
    """Batch load embeddings for a set of node IDs. Returns {node_id: numpy_array}."""
    import numpy as np
    if not node_ids:
        return {}
    full_ids = [nid for nid in node_ids if nid]

    if not full_ids:
        return {}

    ph = ','.join('?' for _ in full_ids)
    rows = conn.execute(
        'SELECT node_id, embedding FROM node_embeddings WHERE node_id IN (%s)' % ph,
        full_ids
    ).fetchall()

    result = {}
    for nid, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        result[nid] = vec
    return result


def _cosine_sim(a, b):
    """Cosine similarity between two numpy arrays."""
    import numpy as np
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _get_edges_strategy_b(brain, node_id, edge_limit=3, query_vec=None,
                          fatigue_state=None):
    """Strategy B: relevance × weight × fatigue discount.

    get_rich_node returns ALL intentional edges. B scores and ranks them.

    query_vec: numpy array (768d) — the embedded query
    fatigue_state: dict {edge_target_id: surface_count} — mutable, updated in place
    """
    from servers.pipeline_contract import get_rich_node
    rich = get_rich_node(brain, node_id)
    if not rich:
        return [], []

    conns = rich.get('connections', [])
    all_available = [c.get('id', '')[:8] for c in conns]

    if not conns or query_vec is None:
        # Fall back to static weight
        conns_sorted = sorted(conns, key=lambda c: c.get('weight', 0), reverse=True)
        shown = [c.get('id', '')[:8] for c in conns_sorted[:edge_limit]]
        return shown, all_available

    # Load embeddings for ALL edge targets (not just top 10)
    target_ids = [c.get('id', '')[:8] for c in conns]
    embeddings = _load_embeddings_for_nodes(brain.conn, target_ids)

    # Score each edge
    scored = []
    for c in conns:
        tid = c.get('id', '')[:8]
        weight = c.get('weight', 0.5)

        # Relevance: cosine sim between query and edge target embedding
        target_vec = embeddings.get(tid)
        if target_vec is not None and len(target_vec) == len(query_vec):
            relevance = max(0, _cosine_sim(query_vec, target_vec))
        else:
            relevance = 0.3  # Default for missing embeddings

        # Fatigue: session rotation
        fatigue_count = 0
        if fatigue_state is not None:
            fatigue_count = fatigue_state.get(tid, 0)
        fatigue_discount = 1.0 / (1.0 + K_EDGE_FATIGUE * fatigue_count)

        score = relevance * weight * fatigue_discount
        scored.append((tid, score, c))

    # Sort by score, pick top N
    scored.sort(key=lambda x: x[1], reverse=True)
    shown = [s[0] for s in scored[:edge_limit]]

    # Update fatigue state for shown edges
    if fatigue_state is not None:
        for tid in shown:
            fatigue_state[tid] = fatigue_state.get(tid, 0) + 1

    return shown, all_available


# ═══════════════════════════════════════════════════════════════
# STRATEGY C: Blended relevance (70% node + 30% description) × weight × fatigue
# ═══════════════════════════════════════════════════════════════

DESC_WEIGHT = 0.3   # How much edge description contributes to relevance
NODE_WEIGHT = 0.7   # How much stored node embedding contributes

def _get_edges_strategy_c(brain, node_id, edge_limit=3, query_vec=None,
                          fatigue_state=None):
    """Strategy C: blended relevance × weight × fatigue.

    relevance = 0.7 * cosine(query, stored_node_embedding)
              + 0.3 * cosine(query, embed(edge_description))

    Falls back to pure node embedding when description is missing.
    """
    import numpy as np
    from servers.pipeline_contract import get_rich_node
    from servers.embedder import embed

    rich = get_rich_node(brain, node_id)
    if not rich:
        return [], []

    conns = rich.get('connections', [])
    all_available = [c.get('id', '')[:8] for c in conns]

    if not conns or query_vec is None:
        conns_sorted = sorted(conns, key=lambda c: c.get('weight', 0), reverse=True)
        shown = [c.get('id', '')[:8] for c in conns_sorted[:edge_limit]]
        return shown, all_available

    # Load stored embeddings for node signal
    target_ids = [c.get('id', '')[:8] for c in conns]
    stored_embeddings = _load_embeddings_for_nodes(brain.conn, target_ids)

    scored = []
    for c in conns:
        tid = c.get('id', '')[:8]
        weight = c.get('weight', 0.5)
        desc = c.get('description', '')

        # Node relevance (always available if embedding exists)
        node_rel = 0.3
        target_vec = stored_embeddings.get(tid)
        if target_vec is not None and len(target_vec) == len(query_vec):
            node_rel = max(0, _cosine_sim(query_vec, target_vec))

        # Description relevance (only when description exists)
        if desc:
            desc_blob = embed(desc)
            if desc_blob is not None:
                desc_vec = np.frombuffer(desc_blob, dtype=np.float32)
                if len(desc_vec) == len(query_vec):
                    desc_rel = max(0, _cosine_sim(query_vec, desc_vec))
                    relevance = NODE_WEIGHT * node_rel + DESC_WEIGHT * desc_rel
                else:
                    relevance = node_rel
            else:
                relevance = node_rel
        else:
            relevance = node_rel

        # Fatigue
        fatigue_count = 0
        if fatigue_state is not None:
            fatigue_count = fatigue_state.get(tid, 0)
        fatigue_discount = 1.0 / (1.0 + K_EDGE_FATIGUE * fatigue_count)

        score = relevance * weight * fatigue_discount
        scored.append((tid, score, c))

    scored.sort(key=lambda x: x[1], reverse=True)
    shown = [s[0] for s in scored[:edge_limit]]

    if fatigue_state is not None:
        for tid in shown:
            fatigue_state[tid] = fatigue_state.get(tid, 0) + 1

    return shown, all_available


# ═══════════════════════════════════════════════════════════════
# STRATEGY D: Relevance-primary, weight as tiebreaker
# NOTE: Weight is currently static (mostly 0.60). When S2 makes weight
# dynamic, restore weight's role in the score formula.
# ═══════════════════════════════════════════════════════════════

WEIGHT_TIEBREAKER = 0.01  # Small enough to never override relevance

def _score_edge_d(relevance, weight, fatigue_count):
    """Score = relevance + (weight × tiebreaker) × fatigue_discount.
    Weight only matters when two edges have equal relevance."""
    fatigue_discount = 1.0 / (1.0 + K_EDGE_FATIGUE * fatigue_count)
    return (relevance + weight * WEIGHT_TIEBREAKER) * fatigue_discount


def _get_edges_strategy_d(brain, node_id, edge_limit=3, query_vec=None,
                          fatigue_state=None, prior_vecs=None):
    """Strategy D: 70% node + 30% description relevance, weight as tiebreaker, 3-msg blend."""
    import numpy as np
    from servers.pipeline_contract import get_rich_node
    from servers.embedder import embed

    # Multi-turn blend
    if query_vec is not None and prior_vecs:
        all_vecs = [query_vec] + prior_vecs
        blended = _blend_query_vecs(all_vecs)
    else:
        blended = query_vec

    rich = get_rich_node(brain, node_id)
    if not rich:
        return [], []

    conns = rich.get('connections', [])
    all_available = [c.get('id', '')[:8] for c in conns]

    if not conns or blended is None:
        conns_sorted = sorted(conns, key=lambda c: c.get('weight', 0), reverse=True)
        shown = [c.get('id', '')[:8] for c in conns_sorted[:edge_limit]]
        return shown, all_available

    # Load stored embeddings
    target_ids = [c.get('id', '')[:8] for c in conns]
    stored_embeddings = _load_embeddings_for_nodes(brain.conn, target_ids)

    scored = []
    for c in conns:
        tid = c.get('id', '')[:8]
        weight = c.get('weight', 0.5)
        desc = c.get('description', '')

        # Node relevance
        node_rel = 0.3
        target_vec = stored_embeddings.get(tid)
        if target_vec is not None and len(target_vec) == len(blended):
            node_rel = max(0, _cosine_sim(blended, target_vec))

        # Description relevance (blended)
        if desc:
            desc_blob = embed(desc)
            if desc_blob is not None:
                desc_vec = np.frombuffer(desc_blob, dtype=np.float32)
                if len(desc_vec) == len(blended):
                    desc_rel = max(0, _cosine_sim(blended, desc_vec))
                    relevance = NODE_WEIGHT * node_rel + DESC_WEIGHT * desc_rel
                else:
                    relevance = node_rel
            else:
                relevance = node_rel
        else:
            relevance = node_rel

        # Fatigue
        fatigue_count = fatigue_state.get(tid, 0) if fatigue_state else 0

        score = _score_edge_d(relevance, weight, fatigue_count)
        scored.append((tid, score, c))

    scored.sort(key=lambda x: x[1], reverse=True)
    shown = [s[0] for s in scored[:edge_limit]]

    if fatigue_state is not None:
        for tid in shown:
            fatigue_state[tid] = fatigue_state.get(tid, 0) + 1

    return shown, all_available


# ═══════════════════════════════════════════════════════════════
# STRATEGY E: Structural embedding — edge_type + target_type + title + description
# ═══════════════════════════════════════════════════════════════

def _get_edges_strategy_e(brain, node_id, edge_limit=3, query_vec=None,
                          fatigue_state=None, prior_vecs=None):
    """Strategy E: embed structural metadata as natural text. Weight as tiebreaker.

    Embed text per edge:
      "{edge_type} {target_type}: {target_title}. {description}"
    Falls back to:
      "{edge_type} {target_type}: {target_title}" when no description.

    Uses multi-turn blend, fatigue, weight as tiebreaker.
    """
    import numpy as np
    from servers.pipeline_contract import get_rich_node
    from servers.embedder import embed

    # Multi-turn blend
    if query_vec is not None and prior_vecs:
        blended = _blend_query_vecs([query_vec] + prior_vecs)
    else:
        blended = query_vec

    rich = get_rich_node(brain, node_id)
    if not rich:
        return [], []

    conns = rich.get('connections', [])
    all_available = [c.get('id', '')[:8] for c in conns]

    if not conns or blended is None:
        conns_sorted = sorted(conns, key=lambda c: c.get('weight', 0), reverse=True)
        return [c.get('id', '')[:8] for c in conns_sorted[:edge_limit]], all_available

    scored = []
    for c in conns:
        tid = c.get('id', '')[:8]
        weight = c.get('weight', 0.5)
        desc = c.get('description', '')
        title = c.get('title', '')
        edge_type = c.get('relation', '') or c.get('edge_type', '') or 'related'
        target_type = c.get('type', '')

        # Build structural embed text
        if desc:
            embed_text = "%s %s: %s. %s" % (edge_type, target_type, title, desc)
        else:
            embed_text = "%s %s: %s" % (edge_type, target_type, title)

        # Embed and score
        relevance = 0.3
        text_blob = embed(embed_text)
        if text_blob is not None:
            text_vec = np.frombuffer(text_blob, dtype=np.float32)
            if len(text_vec) == len(blended):
                relevance = max(0, _cosine_sim(blended, text_vec))

        # Fatigue + weight as tiebreaker
        fatigue_count = fatigue_state.get(tid, 0) if fatigue_state else 0
        score = _score_edge_d(relevance, weight, fatigue_count)
        scored.append((tid, score, c))

    scored.sort(key=lambda x: x[1], reverse=True)
    shown = [s[0] for s in scored[:edge_limit]]

    if fatigue_state is not None:
        for tid in shown:
            fatigue_state[tid] = fatigue_state.get(tid, 0) + 1

    return shown, all_available


# ═══════════════════════════════════════════════════════════════
# STRATEGY F: Description-first — sharpest signal wins
# ═══════════════════════════════════════════════════════════════

def _get_edges_strategy_f(brain, node_id, edge_limit=3, query_vec=None,
                          fatigue_state=None, prior_vecs=None):
    """Strategy F: description-first scoring. No node embedding blending.

    When description exists: embed(description) alone — the edge's own identity.
    When no description: embed(edge_type + target_type + target_title) — sharpest structural fallback.

    The principle: in a flat embedding space, the sharpest signal separates best.
    Node embeddings are broad (title + 1000 chars). Descriptions are narrow (80-120 chars).
    """
    import numpy as np
    from servers.pipeline_contract import get_rich_node
    from servers.embedder import embed

    # Multi-turn blend
    if query_vec is not None and prior_vecs:
        blended = _blend_query_vecs([query_vec] + prior_vecs)
    else:
        blended = query_vec

    rich = get_rich_node(brain, node_id)
    if not rich:
        return [], []

    conns = rich.get('connections', [])
    all_available = [c.get('id', '')[:8] for c in conns]

    if not conns or blended is None:
        conns_sorted = sorted(conns, key=lambda c: c.get('weight', 0), reverse=True)
        return [c.get('id', '')[:8] for c in conns_sorted[:edge_limit]], all_available

    scored = []
    for c in conns:
        tid = c.get('id', '')[:8]
        weight = c.get('weight', 0.5)
        desc = c.get('description', '')
        title = c.get('title', '')
        edge_type = c.get('relation', '') or c.get('edge_type', '') or 'related'
        target_type = c.get('type', '')

        # Description-first: use description when available, structural fallback when not
        if desc:
            embed_text = desc
        else:
            embed_text = "%s %s: %s" % (edge_type, target_type, title)

        relevance = 0.3
        text_blob = embed(embed_text)
        if text_blob is not None:
            text_vec = np.frombuffer(text_blob, dtype=np.float32)
            if len(text_vec) == len(blended):
                relevance = max(0, _cosine_sim(blended, text_vec))

        # Fatigue + weight as tiebreaker
        fatigue_count = fatigue_state.get(tid, 0) if fatigue_state else 0
        score = _score_edge_d(relevance, weight, fatigue_count)
        scored.append((tid, score, c))

    scored.sort(key=lambda x: x[1], reverse=True)
    shown = [s[0] for s in scored[:edge_limit]]

    if fatigue_state is not None:
        for tid in shown:
            fatigue_state[tid] = fatigue_state.get(tid, 0) + 1

    return shown, all_available


# ═══════════════════════════════════════════════════════════════
# MULTI-TURN QUERY BLENDING
# ═══════════════════════════════════════════════════════════════

TURN_WEIGHTS = [0.6, 0.3, 0.1]  # current, previous, two_back

def _blend_query_vecs(query_vecs):
    """Blend multiple query embeddings with decaying weights.

    query_vecs: list of numpy arrays, most recent first.
    Returns: single blended vector (normalized).
    """
    import numpy as np
    if not query_vecs:
        return None
    if len(query_vecs) == 1:
        return query_vecs[0]

    weights = TURN_WEIGHTS[:len(query_vecs)]
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    blended = sum(w * v for w, v in zip(weights, query_vecs))
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended = blended / norm
    return blended


def _get_edges_strategy_b2(brain, node_id, edge_limit=3, query_vec=None,
                           fatigue_state=None, prior_vecs=None):
    """Strategy B2: like B but with multi-turn blended query vector."""
    import numpy as np
    if query_vec is not None and prior_vecs:
        all_vecs = [query_vec] + prior_vecs
        blended = _blend_query_vecs(all_vecs)
    else:
        blended = query_vec
    return _get_edges_strategy_b(brain, node_id, edge_limit,
                                  query_vec=blended, fatigue_state=fatigue_state)


def _get_edges_strategy_c2(brain, node_id, edge_limit=3, query_vec=None,
                           fatigue_state=None, prior_vecs=None):
    """Strategy C2: like C but with multi-turn blended query vector."""
    import numpy as np
    if query_vec is not None and prior_vecs:
        all_vecs = [query_vec] + prior_vecs
        blended = _blend_query_vecs(all_vecs)
    else:
        blended = query_vec
    return _get_edges_strategy_c(brain, node_id, edge_limit,
                                  query_vec=blended, fatigue_state=fatigue_state)


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

def _score_edges(shown_edges, good_edges, bad_edges):
    """Score edge selection quality."""
    shown_set = set(shown_edges)
    good_set = set(good_edges)
    bad_set = set(bad_edges)

    good_shown = shown_set & good_set
    bad_shown = shown_set & bad_set
    total_shown = len(shown_set)

    precision = len(good_shown) / max(total_shown, 1)
    recall = len(good_shown) / max(len(good_set), 1)
    bad_rate = len(bad_shown) / max(total_shown, 1)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "bad_rate": round(bad_rate, 3),
        "good_shown": sorted(good_shown),
        "bad_shown": sorted(bad_shown),
        "neutral_shown": sorted(shown_set - good_set - bad_set),
        "good_missed": sorted(good_set - shown_set),
    }


# ═══════════════════════════════════════════════════════════════
# A/B RUNNERS
# ═══════════════════════════════════════════════════════════════

def _embed_query(brain, query):
    """Embed a query using the brain's embedder. Returns numpy array."""
    import numpy as np
    from servers.embedder import embed
    blob = embed(query)
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def run_single_ab(brain, query_spec, edge_limit=3, query_vec=None, verbose=False):
    """Run a single query through all 5 strategies: A, B, C, B2, C2."""
    node_id = query_spec["target_node"]
    good_edges = query_spec.get("good_edges", [])
    bad_edges = query_spec.get("bad_edges", [])

    # Embed prior queries for multi-turn strategies
    prior_queries = query_spec.get("prior_queries", [])
    prior_vecs = []
    for pq in prior_queries:
        pv = _embed_query(brain, pq)
        if pv is not None:
            prior_vecs.append(pv)

    shown_a, all_a = _get_edges_strategy_a(brain, node_id, edge_limit)
    shown_d, _ = _get_edges_strategy_d(brain, node_id, edge_limit, query_vec=query_vec, prior_vecs=prior_vecs)
    shown_e, _ = _get_edges_strategy_e(brain, node_id, edge_limit, query_vec=query_vec, prior_vecs=prior_vecs)
    shown_f, _ = _get_edges_strategy_f(brain, node_id, edge_limit, query_vec=query_vec, prior_vecs=prior_vecs)

    result = {
        "query": query_spec["query"],
        "category": query_spec["category"],
        "description": query_spec.get("description", ""),
        "target_node": node_id,
        "edge_limit": edge_limit,
        "total_available": len(all_a),
        "has_prior": len(prior_vecs) > 0,
    }
    for label, shown in [("a", shown_a), ("d", shown_d), ("e", shown_e), ("f", shown_f)]:
        result[label] = {"shown": shown, **_score_edges(shown, good_edges, bad_edges)}

    return result


def run_fatigue_ab(brain, query_spec, edge_limit=3, query_vec=None, verbose=False):
    """Run fatigue test through all three strategies."""
    node_id = query_spec["target_node"]
    repeat_count = query_spec.get("repeat_count", 5)
    min_unique = query_spec.get("min_unique_edges", 5)

    def _run_fatigue(strategy_fn, **kwargs):
        fatigue_state = {} if strategy_fn != _get_edges_strategy_a else None
        all_shown = []
        unique = set()
        for i in range(repeat_count):
            fs = fatigue_state if fatigue_state is not None else {}
            shown, all_avail = strategy_fn(brain, node_id, edge_limit,
                                           query_vec=query_vec,
                                           fatigue_state=fs if strategy_fn != _get_edges_strategy_a else None)
            all_shown.append(shown)
            unique.update(shown)
        total = len(all_avail) if all_shown else 0
        return {
            "unique_edges": len(unique),
            "unique_edge_ids": sorted(unique),
            "rotation_rate": round(len(unique) / max(total, 1), 3),
            "passed": len(unique) >= min_unique,
            "identical_runs": sum(1 for s in all_shown if set(s) == set(all_shown[0])),
            "per_run": [sorted(s) for s in all_shown],
        }, total

    a, _ = _run_fatigue(_get_edges_strategy_a)
    # B and C need their own fatigue states
    fatigue_b = {}
    all_shown_b = []
    unique_b = set()
    fatigue_c = {}
    all_shown_c = []
    unique_c = set()
    for i in range(repeat_count):
        sb, avail = _get_edges_strategy_b(brain, node_id, edge_limit,
                                           query_vec=query_vec, fatigue_state=fatigue_b)
        all_shown_b.append(sb)
        unique_b.update(sb)
        sc, _ = _get_edges_strategy_c(brain, node_id, edge_limit,
                                       query_vec=query_vec, fatigue_state=fatigue_c)
        all_shown_c.append(sc)
        unique_c.update(sc)
    total_available = len(avail)

    def _build_fatigue_result(all_shown, unique):
        return {
            "unique_edges": len(unique), "unique_edge_ids": sorted(unique),
            "rotation_rate": round(len(unique) / max(total_available, 1), 3),
            "passed": len(unique) >= min_unique,
            "identical_runs": sum(1 for s in all_shown if set(s) == set(all_shown[0])),
            "per_run": [sorted(s) for s in all_shown],
        }

    b = _build_fatigue_result(all_shown_b, unique_b)
    c = _build_fatigue_result(all_shown_c, unique_c)

    # Strategy D fatigue
    fatigue_d = {}
    all_shown_d = []
    unique_d = set()
    for i in range(repeat_count):
        sd, _ = _get_edges_strategy_d(brain, node_id, edge_limit,
                                       query_vec=query_vec, fatigue_state=fatigue_d)
        all_shown_d.append(sd)
        unique_d.update(sd)
    d = _build_fatigue_result(all_shown_d, unique_d)

    # Strategy E fatigue
    fatigue_e = {}
    all_shown_e = []
    unique_e = set()
    for i in range(repeat_count):
        se, _ = _get_edges_strategy_e(brain, node_id, edge_limit,
                                       query_vec=query_vec, fatigue_state=fatigue_e)
        all_shown_e.append(se)
        unique_e.update(se)
    e = _build_fatigue_result(all_shown_e, unique_e)

    # Strategy F fatigue
    fatigue_f = {}
    all_shown_f = []
    unique_f = set()
    for i in range(repeat_count):
        sf, _ = _get_edges_strategy_f(brain, node_id, edge_limit,
                                       query_vec=query_vec, fatigue_state=fatigue_f)
        all_shown_f.append(sf)
        unique_f.update(sf)
    f = _build_fatigue_result(all_shown_f, unique_f)

    return {
        "query": query_spec["query"], "category": "fatigue",
        "description": query_spec.get("description", ""),
        "target_node": node_id, "repeat_count": repeat_count,
        "edge_limit": edge_limit, "total_available": total_available,
        "min_unique_target": min_unique,
        "a": a, "b": b, "c": c, "d": d, "e": e, "f": f,
    }


def main():
    parser = argparse.ArgumentParser(description="Edge Selection A/B Eval")
    parser.add_argument("--category", help="Filter to specific category")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--edge-limit", type=int, default=3, help="Edges shown per node")
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    queries = EDGE_QUERIES
    if args.category:
        queries = [q for q in queries if q["category"] == args.category]

    STRATEGIES = ['a', 'd', 'e', 'f']
    LABELS = {'a': 'A:weight', 'd': 'D:blend+tie',
              'e': 'E:structural', 'f': 'F:desc-first'}

    print("=" * 110)
    print("EDGE SELECTION EVAL")
    print("  A: Static weight (ORDER BY weight DESC)")
    print("  D: 70%%node+30%%desc, 3-msg blend, weight as tiebreaker")
    print("  E: embed(edge_type + target_type + title + desc), 3-msg blend, weight tiebreaker")
    print("  F: Description-first (desc alone when exists, structural fallback when not)")
    print("  All: 3-msg blend, fatigue K=%.2f, weight as tiebreaker" % K_EDGE_FATIGUE)
    print("  Edge limit: %d | Queries: %d (%s)" % (args.edge_limit, len(queries), args.category or "all"))
    print("=" * 110)
    print()

    with IsolatedBrain() as env:
        single_results = []
        fatigue_results = []

        for q in queries:
            query_vec = _embed_query(env.brain, q["query"])

            if q["category"] == "fatigue":
                r = run_fatigue_ab(env.brain, q, args.edge_limit, query_vec, args.verbose)
                fatigue_results.append(r)

                print("  [fatigue] %s" % q["query"])
                for label in ['a', 'b', 'c']:
                    s = r[label]
                    print("    %s: %d unique / %d avail (%.0f%% rotation) | identical: %d/%d | %s" % (
                        label.upper(), s["unique_edges"], r["total_available"],
                        s["rotation_rate"] * 100, s["identical_runs"],
                        r["repeat_count"], "✓" if s["passed"] else "✗"))
                if args.verbose:
                    for i in range(r["repeat_count"]):
                        print("      Run %d: A=%s  B=%s  C=%s" % (
                            i + 1, r["a"]["per_run"][i], r["b"]["per_run"][i], r["c"]["per_run"][i]))
                print()
            else:
                r = run_single_ab(env.brain, q, args.edge_limit, query_vec, args.verbose)
                single_results.append(r)

                # Find best strategy
                precs = {s: r[s]["precision"] for s in STRATEGIES}
                best = max(precs, key=precs.get)
                if len(set(precs.values())) == 1:
                    best = "tie"
                else:
                    best = best.upper()

                multi_tag = " [3msg]" if r.get("has_prior") else ""
                print("  [%s%s] %s" % (q["category"], multi_tag, q["query"]))
                for s in STRATEGIES:
                    d = r[s]
                    marker = " ←" if s.upper() == best else ""
                    print("    %-10s prec %3.0f%% recall %3.0f%% bad %3.0f%% | %s%s" % (
                        LABELS[s] + ":", d["precision"]*100, d["recall"]*100,
                        d["bad_rate"]*100, d["shown"], marker))
                if args.verbose:
                    for s in STRATEGIES[1:]:
                        if r[s]["good_shown"] != r["a"]["good_shown"]:
                            print("    %s good: %s" % (s.upper(), r[s]["good_shown"]))
                        if r[s]["bad_shown"]:
                            print("    %s bad: %s" % (s.upper(), r[s]["bad_shown"]))
                print()

    # ═══════════════════════════════════════════════════════════════
    # TABLE 1: PER-QUERY A/B COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)

    if single_results:
        print()
        print("┌─ PRECISION BY STRATEGY ─────────────────────────────────────────────────────────────────────────┐")
        print("│ %-30s │  A   │  D   │  E   │  F   │ A Bad│ F Bad│ Best │" % "Query")
        print("├────────────────────────────────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤")
        for r in single_results:
            precs = {s: r[s]["precision"] for s in STRATEGIES}
            best = max(precs, key=precs.get)
            if len(set(precs.values())) == 1:
                best = "="
            else:
                best = best.upper()
            mt = "*" if r.get("has_prior") else " "
            print("│%s%-30s │ %3.0f%% │ %3.0f%% │ %3.0f%% │ %3.0f%% │ %3.0f%% │ %3.0f%% │ %-4s │" % (
                mt, r["query"][:30],
                precs['a'] * 100, precs['d'] * 100, precs['e'] * 100, precs['f'] * 100,
                r["a"]["bad_rate"] * 100, r["f"]["bad_rate"] * 100, best))
        print("│ (* = multi-turn query with prior context)                                                    │")
        print("└────────────────────────────────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 2: FATIGUE A/B COMPARISON
    # ═══════════════════════════════════════════════════════════════
    if fatigue_results:
        print()
        print("┌─ FATIGUE ROTATION A/B/C ──────────────────────────────────────────────────────────────────────────────┐")
        print("│ %-35s │ A Uniq │ B Uniq │ C Uniq │ Avail │ A Rot │ B Rot │ C Rot │" % "Query")
        print("├─────────────────────────────────────┼────────┼────────┼────────┼───────┼───────┼───────┼───────┤")
        for r in fatigue_results:
            a, b, c = r["a"], r["b"], r["c"]
            print("│ %-35s │  %3d   │  %3d   │  %3d   │  %3d  │ %3.0f%%  │ %3.0f%%  │ %3.0f%%  │" % (
                r["query"][:35],
                a["unique_edges"], b["unique_edges"], c["unique_edges"], r["total_available"],
                a["rotation_rate"] * 100, b["rotation_rate"] * 100, c["rotation_rate"] * 100))
        print("└─────────────────────────────────────┴────────┴────────┴────────┴───────┴───────┴───────┴───────┘")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 3: AGGREGATE BY CATEGORY
    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # TABLE 3: AGGREGATE TOTALS
    # ═══════════════════════════════════════════════════════════════
    if single_results:
        n = len(single_results)
        print()
        print("┌─ AGGREGATE TOTALS ──────────────────────────────────────────────────────────────────────────┐")
        print("│              │ Avg Precision │ Avg Recall │ Avg Bad Rate │ Zero-Bad │ Wins vs A │" )
        print("├──────────────┼───────────────┼────────────┼──────────────┼──────────┼───────────┤")
        for label in STRATEGIES:
            avg_p = sum(r[label]["precision"] for r in single_results) / n * 100
            avg_r = sum(r[label]["recall"] for r in single_results) / n * 100
            avg_b = sum(r[label]["bad_rate"] for r in single_results) / n * 100
            clean = sum(1 for r in single_results if r[label]["bad_rate"] == 0)
            wins = sum(1 for r in single_results if r[label]["precision"] > r["a"]["precision"]) if label != 'a' else 0
            losses = sum(1 for r in single_results if r[label]["precision"] < r["a"]["precision"]) if label != 'a' else 0
            wl = "%d/%d" % (wins, losses) if label != 'a' else "base"
            print("│  %-10s  │    %5.1f%%     │   %5.1f%%   │    %5.1f%%    │  %2d/%2d  │  %-7s  │" % (
                LABELS[label], avg_p, avg_r, avg_b, clean, n, wl))
        print("└──────────────┴───────────────┴────────────┴──────────────┴──────────┴───────────┘")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 4: VERDICT
    # ═══════════════════════════════════════════════════════════════
    print()
    def _count_wins(label, baseline='a'):
        wins = sum(1 for r in single_results if r[label]["precision"] > r[baseline]["precision"])
        losses = sum(1 for r in single_results if r[label]["precision"] < r[baseline]["precision"])
        ties = sum(1 for r in single_results if r[label]["precision"] == r[baseline]["precision"])
        bad_better = sum(1 for r in single_results if r[label]["bad_rate"] < r[baseline]["bad_rate"])
        bad_worse = sum(1 for r in single_results if r[label]["bad_rate"] > r[baseline]["bad_rate"])
        return wins, losses, ties, bad_better, bad_worse

    dw, dl, dt, db, dworse = _count_wins('d')
    ew, el, et, eb, eworse = _count_wins('e')
    fw, fl, ft, fb, fworse = _count_wins('f')
    f_vs_d_w, f_vs_d_l, f_vs_d_t, _, _ = _count_wins('f', 'd')
    f_vs_e_w, f_vs_e_l, f_vs_e_t, _, _ = _count_wins('f', 'e')

    print("┌─ VERDICT ───────────────────────────────────────────────────────────────────────────────────┐")
    print("│  D vs A precision:  D wins %d / A wins %d / ties %d  (of %d)" % (dw, dl, dt, len(single_results)))
    print("│  E vs A precision:  E wins %d / A wins %d / ties %d" % (ew, el, et))
    print("│  F vs A precision:  F wins %d / A wins %d / ties %d" % (fw, fl, ft))
    print("│  F vs D precision:  F wins %d / D wins %d / ties %d" % (f_vs_d_w, f_vs_d_l, f_vs_d_t))
    print("│  F vs E precision:  F wins %d / E wins %d / ties %d" % (f_vs_e_w, f_vs_e_l, f_vs_e_t))
    print("│  D vs A bad edges:  D better %d / D worse %d" % (db, dworse))
    print("│  F vs A bad edges:  F better %d / F worse %d" % (fb, fworse))
    if fatigue_results:
        for label in ['a', 'b', 'c']:
            avg_rot = sum(r[label]["rotation_rate"] for r in fatigue_results) / len(fatigue_results)
            passed = sum(1 for r in fatigue_results if r[label]["passed"])
            print("│  %s fatigue:     rotation %.0f%% (%d/%d pass)" % (
                label.upper(), avg_rot * 100, passed, len(fatigue_results)))
    print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 5: C IMPROVEMENTS OVER B (description-aware wins)
    # ═══════════════════════════════════════════════════════════════
    f_over_d = [r for r in single_results if r["f"]["precision"] > r["d"]["precision"]]
    f_under_d = [r for r in single_results if r["f"]["precision"] < r["d"]["precision"]]

    if f_over_d:
        print()
        print("┌─ F IMPROVEMENTS OVER D (description-first wins) ────────────────────────────────────────┐")
        for r in f_over_d:
            d, f = r["d"], r["f"]
            print("│ %s" % r["query"])
            print("│   D showed: %-55s prec: %3.0f%%" % (str(d["shown"]), d["precision"] * 100))
            print("│   F showed: %-55s prec: %3.0f%%" % (str(f["shown"]), f["precision"] * 100))
            f_new = set(f["good_shown"]) - set(d["good_shown"])
            if f_new:
                print("│   F found: %s" % sorted(f_new))
            d_bad_fixed = set(d["bad_shown"]) - set(f["bad_shown"])
            if d_bad_fixed:
                print("│   F eliminated bad: %s" % sorted(d_bad_fixed))
            print("│")
        print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")

    if f_under_d:
        print()
        print("┌─ ⚠ F REGRESSIONS vs D ─────────────────────────────────────────────────────────────────┐")
        for r in f_under_d:
            d, f = r["d"], r["f"]
            print("│ %s" % r["query"])
            print("│   D: prec %3.0f%% %s" % (d["precision"] * 100, d["shown"]))
            print("│   F: prec %3.0f%% %s" % (f["precision"] * 100, f["shown"]))
            print("│")
        print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")

    # Save results
    results_path = ROOT / 'eval' / 'results' / 'edge_selection_ab_latest.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"edge_limit": args.edge_limit, "K_fatigue": K_EDGE_FATIGUE},
        "single_results": single_results,
        "fatigue_results": fatigue_results,
        "summary": {
            **{"%s_avg_precision" % s: round(sum(r[s]["precision"] for r in single_results) / max(len(single_results), 1), 3)
               for s in STRATEGIES},
            **{"%s_avg_bad" % s: round(sum(r[s]["bad_rate"] for r in single_results) / max(len(single_results), 1), 3)
               for s in STRATEGIES},
            "d_vs_a_wins": dw, "e_vs_a_wins": ew, "f_vs_a_wins": fw, "f_vs_d_wins": f_vs_d_w,
        },
    }
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print()
    print("Results saved: %s" % results_path)


if __name__ == "__main__":
    main()
