#!/usr/bin/env python3
"""
Seed Brain — foundational knowledge for every new brain instance.

These are the rules, mental models, and protocols that every brain should
know from birth. They encode hard-won lessons from the brain project's
development — things that took painful sessions to learn.

Usage:
    python3 scripts/seed_brain.py [--db-dir /path/to/brain/]

If --db-dir is not provided, uses resolve-brain-db.sh logic.
"""

import sys
import os
import argparse

# Resolve brain module path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from servers.brain import Brain


# ── Seed Nodes ──────────────────────────────────────────────────────────
# Each entry: (type, title, content, keywords, locked)

SEED_NODES = [
    # ── Brain-Claude Conflict Protocol ──
    (
        "rule",
        "Brain-Claude conflict protocol: STOP, surface to operator, WAIT",
        (
            "When a brain rule blocks or warns against Claude's intended action, "
            "and Claude believes the block is wrong or misapplied:\n"
            "1. STOP immediately — do not retry, workaround, or fight the block.\n"
            "2. SURFACE the disagreement to the operator: \"Brain says X. I think Y because Z. "
            "Who is right?\"\n"
            "3. WAIT for the operator's resolution before proceeding.\n\n"
            "Never fight silently. Never silently comply if you believe the rule is wrong — "
            "that wastes turns and hides signal. The disagreement itself is valuable data.\n\n"
            "Origin: Session where brain's test integrity rule blocked legitimate test updates. "
            "Claude fought the hook 4+ times with different workaround strategies before the "
            "operator noticed and intervened. The operator said: \"ask me and don't argue.\""
        ),
        "conflict escalation protocol brain-claude disagreement operator rule block",
        True,  # locked
    ),

    # ── Brain Identity ──
    (
        "mental_model",
        "Brain is a communicating entity, not infrastructure — attribute decisions to brain, not hooks",
        (
            "The brain speaks to Claude through hook channels (recall, awareness, consciousness, "
            "safety, checkpoints). The hooks are delivery mechanisms. The brain is the intelligence.\n\n"
            "Correct: \"The brain blocked this command because...\"\n"
            "Wrong: \"The hook blocked this command because...\"\n\n"
            "All brain output is wrapped in [BRAIN]...[/BRAIN] tags to mark authorship. "
            "When referencing brain-authored rules, decisions, or warnings, attribute them "
            "to the brain, not to the hook system."
        ),
        "brain identity attribution hooks communication mental-model",
        True,  # locked
    ),

    # ── Test Integrity ──
    (
        "rule",
        "Test integrity: when a test fails, STOP and ask the operator",
        (
            "When a test fails, do NOT change the test OR the code to make it pass.\n"
            "1. STOP — do not change anything.\n"
            "2. REPORT: what the test expected vs what the code returned.\n"
            "3. ASK: \"Is the test wrong, or does the code have a bug?\"\n"
            "4. WAIT for the answer.\n\n"
            "This applies in BOTH directions:\n"
            "- Do NOT weaken the test (changing assertEqual to assertGreater, removing assertions).\n"
            "- Do NOT \"fix\" the code to satisfy a test that might be wrong.\n\n"
            "DISTINCTION — updating vs weakening:\n"
            "- UPDATING: Changing expected values because the implementation intentionally changed "
            "(e.g., you replaced a stub with real logic, so the output format changes). "
            "Ask: Was this change planned? Does the new value reflect correct behavior?\n"
            "- WEAKENING: Making assertions less strict to hide a bug (e.g., assertEqual → "
            "assertGreater, adding try/except around assertions). This is never acceptable."
        ),
        "test integrity assertions regression stop-and-ask updating weakening",
        True,  # locked
    ),

    # ── Eval Design ──
    (
        "lesson",
        "Design evals from desired behavior change, not from code change",
        (
            "When testing whether a change works, ask: \"What behavior should be different?\" "
            "not \"What code did I modify?\"\n\n"
            "Always include negative/noise test cases — positive cases often saturate "
            "(the model does the right thing regardless of format). Noise cases test "
            "discrimination ability, which is where real differences surface.\n\n"
            "Example: Testing [BRAIN] tags. 4 relevant-recall scenarios showed identical results. "
            "The only differentiator was the irrelevant-recall scenario — plain text caused "
            "false encoding (10%), [BRAIN] tags eliminated it (0%)."
        ),
        "eval methodology testing noise-cases behavior-driven negative-testing",
        False,  # not locked — this is a lesson, not a rule
    ),

    # ── Brain Documents Philosophy ──
    (
        "rule",
        "Brain is associative memory, not a document store — use markdown for formal content",
        (
            "Don't encode formal plans, task lists, or specs as brain nodes. Those belong "
            "in markdown files in the repo.\n\n"
            "Brain nodes should POINT TO documents, not duplicate them.\n"
            "Example: a brain node says 'refactoring targets are in REFACTORING.md — top "
            "priority is X because it blocks Y.' The node holds the WHY and the POINTER. "
            "The document holds the WHAT."
        ),
        "brain documents philosophy associative-memory markdown division-of-labor",
        True,  # locked
    ),
]

# ── Seed Connections ───────────────────────────────────────────────────
# (source_title_fragment, target_title_fragment, relation, weight)

SEED_CONNECTIONS = [
    ("conflict protocol", "communicating entity", "requires_understanding_of", 0.9),
    ("conflict protocol", "test integrity", "example_of", 0.8),
    ("Design evals from", "conflict protocol", "informed_by", 0.7),
]


def seed_brain(db_dir):
    """Insert seed nodes into a brain, skipping duplicates."""
    db_path = os.path.join(db_dir, "brain.db")
    brain = Brain(db_path)

    created = 0
    skipped = 0

    for node_type, title, content, keywords, locked in SEED_NODES:
        # Check for existing node with exact title match
        exact = brain.conn.execute(
            "SELECT id FROM nodes WHERE title = ?", (title,)
        ).fetchone()
        if exact:
            print("  SKIP (exists): %s" % title[:60])
            skipped += 1
            continue

        brain.remember(
            type=node_type,
            title=title,
            content=content,
            keywords=keywords,
            locked=locked,
        )
        created += 1
        print("  SEED: %s" % title[:60])

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
    # 1. Env var override
    env_dir = os.environ.get("BRAIN_DB_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir

    # 2. Cowork mount
    sessions_base = "/sessions"
    if os.path.isdir(sessions_base):
        for d in os.listdir(sessions_base):
            mount_path = os.path.join(sessions_base, d, "mnt", "AgentsContext", "brain")
            if os.path.isdir(mount_path):
                return mount_path

    # 3. Local (typically symlink to Google Drive)
    home = os.path.expanduser("~")
    local_path = os.path.join(home, "AgentsContext", "brain")
    if os.path.isdir(local_path):
        return local_path

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed a new brain with foundational knowledge")
    parser.add_argument("--db-dir", help="Path to brain DB directory")
    args = parser.parse_args()

    db_dir = args.db_dir or resolve_db_dir()
    if not db_dir:
        print("ERROR: Cannot resolve brain DB directory. Use --db-dir or set BRAIN_DB_DIR.")
        sys.exit(1)

    print("Seeding brain at: %s" % db_dir)
    seed_brain(db_dir)
