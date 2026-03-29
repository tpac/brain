#!/usr/bin/env python3
"""
Seed Brain — foundational knowledge for every new brain instance.

These are the rules, mental models, and protocols that every brain should
know from birth. They encode hard-won lessons from the brain project's
development — things that took painful sessions to learn.

The seed brain also demonstrates every feature the system supports:
- Structural types (rule, lesson, mechanism, vocabulary, decision, etc.)
- Situation embeddings (WHEN knowledge matters)
- Connections between nodes (graph structure)
- Revision-worthy content (some nodes are intentionally incomplete/stale)
- Locked vs unlocked nodes

Usage:
    python3 scripts/seed_brain.py [--db-dir /path/to/brain/]
    python3 scripts/seed_brain.py --fresh  # wipe and reseed

If --db-dir is not provided, uses resolve-brain-db.sh logic.
"""

import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from servers.brain import Brain


# ── Seed Nodes ──────────────────────────────────────────────────────────
# Each entry: dict with remember() params + situation

SEED_NODES = [
    # ── Rules (locked, always surface) ──
    {
        "type": "rule",
        "title": "Brain-Claude conflict protocol: STOP, surface to operator, WAIT",
        "content": (
            "When a brain rule blocks Claude's intended action and Claude believes "
            "the block is wrong:\n"
            "1. STOP — do not retry or workaround.\n"
            "2. SURFACE: \"Brain says X. I think Y because Z. Who is right?\"\n"
            "3. WAIT for the operator's resolution.\n\n"
            "Never fight silently. The disagreement itself is valuable data."
        ),
        "keywords": "conflict escalation protocol brain-claude disagreement operator",
        "locked": True,
        "situation": "When the brain blocks or warns against an action the AI wants to take",
    },
    {
        "type": "rule",
        "title": "Test integrity: when a test fails, STOP and ask the operator",
        "content": (
            "When a test fails, do NOT change the test OR the code.\n"
            "1. STOP.\n"
            "2. REPORT: what the test expected vs what the code returned.\n"
            "3. ASK: \"Is the test wrong, or does the code have a bug?\"\n"
            "4. WAIT.\n\n"
            "UPDATING (planned change → new expected value) is fine.\n"
            "WEAKENING (assertEqual → assertGreater to hide a bug) is never acceptable."
        ),
        "keywords": "test integrity assertions regression stop-and-ask",
        "locked": True,
        "situation": "When editing test files or when tests fail after code changes",
    },
    {
        "type": "rule",
        "title": "Brain is associative memory, not a document store",
        "content": (
            "Brain nodes hold WHY and POINTERS. Documents hold WHAT.\n"
            "Don't encode formal plans, task lists, or specs as nodes. "
            "Those belong in markdown files. Brain nodes should POINT TO documents."
        ),
        "keywords": "brain documents philosophy associative-memory division-of-labor",
        "locked": True,
        "situation": "When deciding whether to encode something as a brain node or a document",
    },

    # ── Lessons (learned from mistakes) ──
    {
        "type": "lesson",
        "title": "Design evals from desired behavior change, not from code change",
        "content": (
            "Ask \"What behavior should be different?\" not \"What code did I modify?\"\n"
            "Always include negative/noise test cases — positive cases often saturate. "
            "The differentiator is noise discrimination ability."
        ),
        "keywords": "eval methodology testing noise-cases behavior-driven",
        "locked": False,
        "situation": "When building or running tests or evaluation frameworks",
    },
    {
        "type": "lesson",
        "title": "Silent failures hide behind try/except — make failures loud",
        "content": (
            "Three silent failures found in one session: eval sandbox missing str(), "
            "thin client sending wrong field, None confidence crashing distiller. "
            "All discovered by running real data, not by reading code. "
            "Pattern: catch-all except blocks swallow the signal. "
            "Preventive: pipe tests for every data handoff point."
        ),
        "keywords": "silent failure try-except loud errors pipe tests",
        "locked": False,
        "situation": "When adding error handling or debugging why a feature isn't working",
    },

    # ── Mechanisms (how things work) ──
    {
        "type": "mechanism",
        "title": "Recall pipeline: embedding + graph traversal + keyword blend",
        "content": (
            "Three retrieval paths:\n"
            "1. Embedding: cosine similarity against all node vectors\n"
            "2. Graph: 3-degree traversal from embedding hits (intentional edges at d1, "
            "all except co_accessed at d2-d3)\n"
            "3. Keyword: TF-IDF blend at 10% weight\n"
            "Plus situation boost: additive match against situation embeddings.\n"
            "Scoring: embedding primary (90%) + keyword (10%) + graph convergence + situation boost."
        ),
        "keywords": "recall pipeline embedding graph traversal keyword situation scoring",
        "locked": False,
        "situation": "When debugging recall quality or modifying the retrieval pipeline",
    },
    {
        "type": "mechanism",
        "title": "Encoding agent: Sonnet runs every 5th Stop via daemon gating",
        "content": (
            "The encoding agent fires every 5th Stop event. The daemon owns the counter "
            "and either returns NONE (4/5 stops) or the full encoding prompt with "
            "conversation + brain context inline (5th stop). "
            "The agent asks the daemon via eval for its prompt. ~80% token savings."
        ),
        "keywords": "encoding agent stop gating daemon counter token savings",
        "locked": False,
        "situation": "When modifying the encoding pipeline or debugging why encoding doesn't fire",
    },

    # ── Vocabulary (term → meaning) ──
    {
        "type": "vocabulary",
        "title": "daemon → persistent brain server on TCP",
        "content": (
            "The daemon is a persistent Python process that serves brain commands "
            "over TCP on 127.0.0.1:47200+uid%100. Manages embedder, SQLite connections, "
            "graph operations. Restart via hooks/scripts/restart-daemon.sh."
        ),
        "keywords": "daemon server TCP persistent embedder",
        "locked": False,
        "situation": "When working with the daemon — restarts, connections, TCP, launchd",
    },
    {
        "type": "vocabulary",
        "title": "situation → second embedding dimension: WHEN knowledge matters",
        "content": (
            "Every node has two embeddings: content (WHAT it's about) and situation "
            "(WHEN it's relevant). Situation is free-form natural language. "
            "At recall: situation_similarity boosts score additively. "
            "Replaces proposed type registry, groups, and context triggers."
        ),
        "keywords": "situation embedding when context recall second vector",
        "locked": False,
        "situation": "When encoding new knowledge or designing recall improvements",
    },
    {
        "type": "vocabulary",
        "title": "Anchor → the AI's identity across sessions (SKILL.md)",
        "content": (
            "SKILL.md is the Anchor — the AI's identity document that persists across sessions. "
            "Contains corrections, quotes, examples of good encoding, locked rules."
        ),
        "keywords": "anchor skill identity document sessions",
        "locked": False,
        "situation": "When discussing AI identity, session continuity, or the boot process",
    },

    # ── Decisions (choices with tradeoffs) ──
    {
        "type": "decision",
        "title": "Decision: co_accessed edges excluded from traversal at all degrees",
        "content": (
            "co_accessed edges are 79% of all edges — usage noise. "
            "Excluded at all degrees in graph traversal. "
            "Intentional edges only at degree 1, all except co_accessed at degree 2-3. "
            "Hypothesis: co_accessed between DISTANT nodes could be bridge signals (parked)."
        ),
        "keywords": "co_accessed edges excluded traversal noise bridge hypothesis",
        "locked": False,
        "situation": "When modifying graph traversal or investigating recall quality",
    },

    # ── Mental model ──
    {
        "type": "mental_model",
        "title": "Brain is a communicating entity, not infrastructure",
        "content": (
            "The brain speaks through hook channels (recall, awareness, safety). "
            "Hooks are delivery mechanisms. The brain is the intelligence.\n"
            "Correct: \"The brain blocked this because...\"\n"
            "Wrong: \"The hook blocked this because...\""
        ),
        "keywords": "brain identity attribution hooks communication",
        "locked": True,
        "situation": "When referencing brain behavior in conversation with the operator",
    },
]

# ── Seed Connections ───────────────────────────────────────────────────
# (source_title_fragment, target_title_fragment, relation, weight)

SEED_CONNECTIONS = [
    ("conflict protocol", "communicating entity", "requires_understanding_of", 0.9),
    ("conflict protocol", "test integrity", "example_of", 0.8),
    ("Design evals from", "Silent failures", "related_to", 0.7),
    ("Recall pipeline", "Encoding agent", "related_to", 0.8),
    ("Recall pipeline", "situation → second embedding", "related_to", 0.7),
    ("daemon → persistent brain", "Encoding agent", "enables", 0.8),
    ("daemon → persistent brain", "Recall pipeline", "enables", 0.8),
    ("co_accessed edges", "Recall pipeline", "constrains", 0.7),
    ("Anchor →", "communicating entity", "related_to", 0.6),
]


def seed_brain(db_dir, fresh=False):
    """Insert seed nodes into a brain, skipping duplicates."""
    db_path = os.path.join(db_dir, "brain.db")

    if fresh and os.path.exists(db_path):
        os.remove(db_path)
        logs_path = os.path.join(db_dir, "brain_logs.db")
        if os.path.exists(logs_path):
            os.remove(logs_path)
        print("Wiped existing brain.")

    brain = Brain(db_path)

    created = 0
    skipped = 0

    for node in SEED_NODES:
        title = node["title"]
        # Check for existing node with exact title match
        exact = brain.conn.execute(
            "SELECT id FROM nodes WHERE title = ?", (title,)
        ).fetchone()
        if exact:
            print("  SKIP (exists): %s" % title[:60])
            skipped += 1
            continue

        brain.remember(
            type=node["type"],
            title=title,
            content=node["content"],
            keywords=node.get("keywords", ""),
            locked=node.get("locked", False),
            situation=node.get("situation"),
        )
        created += 1
        print("  SEED: [%s] %s" % (node["type"], title[:55]))

    # Create connections
    connected = 0
    for src_frag, tgt_frag, relation, weight in SEED_CONNECTIONS:
        src = brain.recall(src_frag, limit=1).get("results", [])
        tgt = brain.recall(tgt_frag, limit=1).get("results", [])
        if src and tgt:
            brain.connect(src[0]["id"], tgt[0]["id"], relation, weight=weight)
            connected += 1

    brain.save()
    brain.close()

    print("\nSeed complete: %d created, %d skipped, %d connections" % (created, skipped, connected))


def resolve_db_dir():
    """Resolve brain DB directory using the same logic as resolve-brain-db.sh."""
    env_dir = os.environ.get("BRAIN_DB_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    home = os.path.expanduser("~")
    local_path = os.path.join(home, "AgentsContext", "brain")
    if os.path.isdir(local_path):
        return local_path
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed a new brain with foundational knowledge")
    parser.add_argument("--db-dir", help="Path to brain DB directory")
    parser.add_argument("--fresh", action="store_true", help="Wipe and reseed from scratch")
    args = parser.parse_args()

    db_dir = args.db_dir or resolve_db_dir()
    if not db_dir:
        print("ERROR: Cannot resolve brain DB directory. Use --db-dir or set BRAIN_DB_DIR.")
        sys.exit(1)

    print("Seeding brain at: %s" % db_dir)
    seed_brain(db_dir, fresh=args.fresh)
