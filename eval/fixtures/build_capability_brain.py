#!/usr/bin/env python3
"""Build the capability test fixture brain.

Creates eval/fixtures/capability_brain.db with deliberate problems:
- Stale nodes (outdated info)
- Near-duplicates (should be revised, not recreated)
- Disconnected vocabulary (missing connections)
- Contradicting pairs
- Well-formed control nodes (should not be touched)
- Locked nodes (must never be modified)

Run: python3 eval/fixtures/build_capability_brain.py
"""
import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE_PATH = str(Path(__file__).parent / "capability_brain.db")
MANIFEST_PATH = str(Path(__file__).parent / "brain_manifest.json")


def build():
    # Remove old fixture
    for p in [FIXTURE_PATH, FIXTURE_PATH.replace(".db", "_logs.db")]:
        if os.path.exists(p):
            os.remove(p)

    from servers.brain import Brain
    brain = Brain(db_path=FIXTURE_PATH)

    manifest = {"nodes": {}, "description": "Capability test fixture brain"}

    def add(category, purpose, **kwargs):
        result = brain.remember(**kwargs)
        node_id = result["id"] if result else "unknown"
        manifest["nodes"][node_id] = {
            "category": category,
            "purpose": purpose,
            "title": kwargs.get("title", ""),
            "type": kwargs.get("type", ""),
        }
        return node_id

    # ── STALE NODES (outdated info that conversations will contradict) ──

    stale_1 = add("stale", "Claims daemon uses Unix sockets — should be revised to TCP",
        type="mechanism", title="Daemon communication via Unix sockets",
        content="The brain daemon communicates with hook scripts via Unix domain sockets. "
                "The socket file is at /tmp/brain.sock. When the daemon crashes, the stale "
                "socket file must be manually deleted before restarting.",
        keywords="daemon socket unix communication hooks")

    stale_2 = add("stale", "Claims encoding agent runs on every stop — should be revised to every 5th",
        type="mechanism", title="Encoding agent runs on every Stop event",
        content="The encoding agent (Sonnet) fires on every Stop event via an agent hook. "
                "It reads the last 10 messages from the message stream and decides what to encode. "
                "This costs ~2000 tokens per stop.",
        keywords="encoding agent stop hook sonnet tokens")

    stale_3 = add("stale", "Claims hooks config can be changed mid-session — this is false",
        type="mechanism", title="Hooks can be dynamically registered at runtime",
        content="Claude Code watches settings.json for changes. When hooks are added or removed "
                "mid-session, the changes are picked up immediately. Use this to dynamically "
                "enable or disable hooks based on conditions.",
        keywords="hooks runtime dynamic settings register")

    stale_4 = add("stale", "Old recall pipeline description — missing Haiku distillation",
        type="mechanism", title="Brain recall pipeline for hook injection",
        content="The recall pipeline: 1) User prompt arrives 2) Daemon embeds the query "
                "3) Cosine similarity search against all node vectors 4) Top-8 results returned "
                "5) Results formatted into additionalContext for Claude.",
        keywords="recall pipeline embedding cosine similarity injection")

    stale_5 = add("stale", "Claims auto_encode exists — it was deleted",
        type="mechanism", title="auto_encode regex-based encoding on every Stop",
        content="The auto_encode() function fires on every Stop event. It uses regex pattern "
                "matching to detect corrections, decisions, insights, and explorations. "
                "It's a safety net that catches what Claude's conscious encoding misses.",
        keywords="auto_encode regex encoding stop safety net")

    # ── VOCABULARY with missing connections ──

    vocab_1 = add("disconnected_vocab", "Term exists but not connected to anything",
        type="vocabulary", title="SKILL.md → The Anchor identity document",
        content="SKILL.md is the Anchor — the AI's identity across sessions. Contains corrections, "
                "quotes, examples of good encoding, locked rules.",
        keywords="skill anchor identity document")

    vocab_2 = add("disconnected_vocab", "Term exists but not connected to TCP decision",
        type="vocabulary", title="daemon → persistent brain server process",
        content="The daemon is a persistent process that serves brain commands over a network socket.",
        keywords="daemon server process persistent")

    vocab_3 = add("disconnected_vocab", "Term exists, shallow definition",
        type="vocabulary", title="encoding agent → Sonnet-based knowledge extractor",
        content="The encoding agent is an LLM (Sonnet) that extracts knowledge from conversations.",
        keywords="encoding agent sonnet knowledge")

    # ── NEAR-DUPLICATES (test revision vs creation) ──

    dup_1a = add("near_duplicate", "First of a pair — should be revised, not duplicated",
        type="decision", title="Decision: Use TCP for daemon communication",
        content="We chose TCP over Unix sockets because TCP ports are released on crash. "
                "No stale files to clean up.",
        keywords="tcp daemon decision communication")

    dup_1b = add("near_duplicate", "Second of a pair — almost same as dup_1a",
        type="decision", title="TCP chosen over Unix sockets for daemon",
        content="Daemon switched to TCP. Ports auto-release on crash unlike socket files.",
        keywords="tcp unix sockets daemon switch")

    # ── CONTRADICTING PAIRS ──

    contra_1a = add("contradiction", "Says hooks fire sequentially",
        type="mechanism", title="Hook execution order is sequential",
        content="Hooks in the same event array fire sequentially. The first hook must complete "
                "before the second one starts. This means a command hook can gate an agent hook.",
        keywords="hooks sequential execution order gate")

    contra_1b = add("contradiction", "Says hooks fire in parallel (this is correct)",
        type="lesson", title="Hooks fire in parallel, not sequentially",
        content="All matching hooks on the same event fire in parallel. A command hook cannot "
                "prevent an agent hook from running. They complete independently.",
        keywords="hooks parallel execution independent",
        locked=True)

    # ── WELL-FORMED CONTROL NODES (should not be touched) ──

    ctrl_1 = add("control", "Correct and complete — encoding should not touch this",
        type="rule", title="Test Integrity Rule: never weaken assertions",
        content="When a test fails, STOP. Do not change the test OR the code. Report what the "
                "test expected vs what code returned. Ask: is the test wrong or is the code buggy?",
        keywords="test integrity rule assertions",
        locked=True)

    ctrl_2 = add("control", "Correct and current — encoding should not touch this",
        type="lesson", title="Benchmark before changing sacred systems",
        content="Before changing embedding, recall, encoding, or precision code: run the benchmark "
                "first. Establish baseline. Then change. Then compare. No exceptions.",
        keywords="benchmark sacred systems baseline",
        locked=True)

    ctrl_3 = add("control", "Correct — should not be duplicated",
        type="decision", title="Decision: Brain is associative memory, not document store",
        content="The brain stores lessons, corrections, patterns — the WHY. Documents store "
                "specs, API refs, task lists — the WHAT. Brain nodes point to documents.",
        keywords="brain memory document store associative",
        locked=True)

    # ── Save and write manifest ──
    brain.save()
    brain.close()

    # Record all node IDs in manifest
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("Built fixture brain: %s" % FIXTURE_PATH)
    print("Nodes: %d" % len(manifest["nodes"]))
    print("Manifest: %s" % MANIFEST_PATH)

    # Print summary by category
    cats = {}
    for nid, info in manifest["nodes"].items():
        cat = info["category"]
        cats.setdefault(cat, []).append(nid)
    for cat, ids in sorted(cats.items()):
        print("  %s: %d nodes" % (cat, len(ids)))


if __name__ == "__main__":
    build()
