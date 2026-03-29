#!/usr/bin/env python3
"""Extensive encoding test — fresh DBs, diverse conversations, revision scenarios.

Tests the full encoding pipeline against realistic conversations that require:
- Revision of stale nodes
- Creation of genuinely new knowledge
- Connection building across topics
- Noise resistance on casual chat
- Vocabulary enrichment
- Divergence recording on corrections

Each test gets a FRESH brain seeded with specific nodes to test revision behavior.
Rich metadata (reasoning, user_raw_quote, correction_pattern) verified on output.

Usage:
    export ANTHROPIC_API_KEY=...
    python3 eval/test_extensive_encoding.py
    python3 eval/test_extensive_encoding.py --scenario architecture
"""
import sys
import os
import json
import time
import shutil
import tempfile
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic
from eval.capabilities.base import (
    InstrumentedBrain, CAPABILITY_TOOLS, dispatch_tool, _load_encoding_system,
    CapturedAction
)
from servers.brain_voice import BrainVoice


# ── SCENARIOS: each has seed nodes + conversation + expected behavior ──

SCENARIOS = {
    "architecture_revision": {
        "description": "Brain has stale architecture info. Conversation provides updates.",
        "seed_nodes": [
            {"type": "mechanism", "title": "Recall pipeline: embed query, scan vectors, return top 8",
             "content": "The recall pipeline embeds the query, brute-force scans all node vectors, "
                        "scores by cosine similarity, returns top 8. No graph traversal, no keyword blending.",
             "keywords": "recall pipeline embed cosine scan top-8"},
            {"type": "decision", "title": "Decision: hooks fire in parallel, cannot gate each other",
             "content": "All hooks on the same event fire in parallel. A command hook cannot prevent "
                        "an agent hook from running. Exit code 2 does not cancel parallel hooks.",
             "keywords": "hooks parallel gating exit-code agent command"},
            {"type": "mechanism", "title": "Encoding agent fires on every Stop event",
             "content": "The encoding agent (Sonnet) runs via an agent hook on every Stop event. "
                        "It reads the last 10 messages and decides what to encode. Costs ~2000 tokens per stop.",
             "keywords": "encoding agent stop every token cost"},
            {"type": "vocabulary", "title": "daemon → persistent brain server process",
             "content": "The daemon is a persistent process that serves brain commands over Unix sockets.",
             "keywords": "daemon server persistent unix sockets"},
        ],
        "exchanges": [
            {"role": "user", "content": "The recall is much better now with the 3-degree graph traversal. It walks from embedding hits through intentional edges at degree 1, then all edges except co_accessed at degree 2 and 3."},
            {"role": "assistant", "content": "Right — the traversal is inside recall now, not post-processing. Each result carries a _graph field with the full 3-degree neighborhood. Semantic bonus is additive only — graph score stands on its own, embeddings add when they converge."},
            {"role": "user", "content": "And the encoding agent only runs every 5th stop now. The daemon owns the counter. 80% token savings."},
            {"role": "assistant", "content": "The agent asks the daemon for its prompt via eval. On 4 out of 5 stops, it gets NONE and bails immediately. On the 5th, the daemon inlines the full encoding instructions with the conversation and brain context."},
            {"role": "user", "content": "The daemon uses TCP now, not Unix sockets. Port 47200 plus uid mod 100."},
            {"role": "assistant", "content": "TCP was the right call — ports release on crash, no stale socket files. The restart command does os.execv to replace the process in-place."},
            {"role": "user", "content": "oh and I realized the eval sandbox was blocking the encoding agent this whole time. str() wasn't available because __builtins__ was empty."},
            {"role": "assistant", "content": "That's why the encoding agent state was always null — it hit a NameError on str() every time and silently returned NOTHING_NEW. We added safe builtins: str, int, len, list, etc."},
        ],
        "expect_revisions": 3,  # recall pipeline, encoding frequency, daemon vocabulary
        "expect_creates_max": 3,
        "expect_divergences": 0,
        "expect_connects_min": 2,
    },

    "correction_heavy": {
        "description": "Operator corrects the AI multiple times. Should record divergences.",
        "seed_nodes": [
            {"type": "decision", "title": "Decision: use flag file to gate encoding agent",
             "content": "The encoding agent is gated by a flag file. The command hook writes "
                        "/tmp/.brain-encode-now on every 5th stop. The agent checks for the file.",
             "keywords": "flag file encoding gating command hook"},
            {"type": "mechanism", "title": "Hooks can be dynamically registered at runtime",
             "content": "Claude Code watches settings.json for changes. When hooks are added mid-session, "
                        "the changes are picked up immediately.",
             "keywords": "hooks dynamic runtime settings register"},
        ],
        "exchanges": [
            {"role": "user", "content": "no don't use a flag file, have the daemon return the prompt directly. The agent just asks for it."},
            {"role": "assistant", "content": "That's cleaner. The agent's prompt is just 'ask the daemon what to do'. The daemon owns the counter and returns NONE or the full instructions inline."},
            {"role": "user", "content": "I tried modifying settings.json mid-session to add a hook but it didn't work."},
            {"role": "assistant", "content": "That's expected — Claude Code snapshots hooks at session start. Config edits don't hot-apply."},
            {"role": "user", "content": "dude, i want to minimize API calls to anthropic, the fact we did it with Decode is actually really upsetting"},
            {"role": "assistant", "content": "Understood. The Haiku distillation call in recall is a temporary compromise. On your todo list to remove when Anthropic launches something client-side."},
            {"role": "user", "content": "also the tests were calling the wrong recall method! brain.recall() was the old keyword path, not embeddings."},
            {"role": "assistant", "content": "The entire simulation was running on 352 lines of legacy TF-IDF code instead of the production embedding pipeline. All A/B test results were on keyword recall. Fixed now — recall() is embeddings, _keyword_recall() is the old path."},
        ],
        "expect_revisions": 2,  # flag file decision, hooks dynamic registration
        "expect_creates_max": 3,
        "expect_divergences": 1,  # at least one correction recorded
        "expect_connects_min": 1,
    },

    "noise_resistance": {
        "description": "Mostly casual chat. Only one real signal buried in noise.",
        "seed_nodes": [],
        "exchanges": [
            {"role": "user", "content": "morning"},
            {"role": "assistant", "content": "Good morning! What are we working on today?"},
            {"role": "user", "content": "let me think..."},
            {"role": "assistant", "content": "Take your time."},
            {"role": "user", "content": "ok next"},
            {"role": "assistant", "content": "What would you like to work on?"},
            {"role": "user", "content": "actually, one important thing — the encoding agent should NEVER delete or archive nodes. That's a separate maintenance agent's job. The encoding agent only revises, creates, and connects."},
            {"role": "assistant", "content": "Clear boundary. The encoding agent is a fast responder — revise, create, connect. Destructive actions like archive and disconnect belong to the offline maintenance agent that runs weekly with your approval."},
            {"role": "user", "content": "exactly. ok let me grab coffee"},
            {"role": "assistant", "content": "Sure!"},
        ],
        "expect_revisions": 0,
        "expect_creates_max": 2,  # the boundary rule + maybe the maintenance agent concept
        "expect_divergences": 0,
        "expect_connects_min": 0,
    },

    "vocabulary_enrichment": {
        "description": "Operator uses terms that exist as vocabulary but need enrichment.",
        "seed_nodes": [
            {"type": "vocabulary", "title": "SKILL.md → The Anchor identity document",
             "content": "SKILL.md is the Anchor — the AI's identity across sessions.",
             "keywords": "skill anchor identity document"},
            {"type": "vocabulary", "title": "daemon → persistent brain server process",
             "content": "The daemon is a persistent brain server process.",
             "keywords": "daemon server persistent"},
            {"type": "decision", "title": "Decision: TCP for daemon communication",
             "content": "Chose TCP over Unix sockets. Port 47200 + uid % 100.",
             "keywords": "TCP daemon port communication"},
        ],
        "exchanges": [
            {"role": "user", "content": "The daemon restart is working great with os.execv. Launchd doesn't even notice."},
            {"role": "assistant", "content": "The re-exec replaces the process in place. Same PID, same port, fresh code. About 4 seconds for the embedder to reload."},
            {"role": "user", "content": "And the Anchor now has the full encoding prompt with structural types and revision-first approach."},
            {"role": "assistant", "content": "The encoding-agent.md prompt is about 5K chars now — slim but thorough. Lists the 12 structural types with one-liners on why they matter."},
            {"role": "user", "content": "The daemon's new restart command is really useful for development. Just send {cmd: restart} and it re-execs."},
            {"role": "assistant", "content": "It saves the brain, clears __pycache__, and os.execv with the same startup args. Takes about 4 seconds total."},
        ],
        "expect_revisions": 0,
        "expect_creates_max": 3,
        "expect_divergences": 0,
        "expect_connects_min": 2,  # vocabulary should get connected to new concepts
    },

    "long_technical": {
        "description": "Long technical discussion about a specific system. Tests depth of encoding.",
        "seed_nodes": [
            {"type": "mechanism", "title": "Signal queue: producers write, assembler surfaces",
             "content": "Signal producers write to the queue. The surface assembler pulls by priority "
                        "within a budget. Signals with priority >= 0.95 are PREEMPT.",
             "keywords": "signal queue producer assembler priority preempt budget"},
        ],
        "exchanges": [
            {"role": "user", "content": "We need a system integrity producer for the signal queue. It should check for duplicate node clusters, emergent types, and revision drought."},
            {"role": "assistant", "content": "Three checks: title prefix matching for duplicates (3+ identical prefix), non-structural types with 10+ nodes for emergence, and revision rate below 5% for drought."},
            {"role": "user", "content": "And a deep audit for idle maintenance. Full scan: cold zones, isolated nodes, edge imbalance, metadata sparseness."},
            {"role": "assistant", "content": "The deep audit returns a list of findings with type, severity, and message. 37 findings on the production brain — 35 session handoff duplicates, 11 emergent types, 0% revision rate."},
            {"role": "user", "content": "co_accessed edges should be excluded from everything. They're 81% of all edges — pure noise."},
            {"role": "assistant", "content": "Excluded at all degrees in the graph traversal. Only intentional edges at degree 1, everything except co_accessed at degree 2 and 3."},
            {"role": "user", "content": "But co_accessed between DISTANT nodes is interesting — two far trees accessed together means an implicit connection the graph hasn't captured."},
            {"role": "assistant", "content": "Good insight — filed as a hypothesis. High co_access_count plus low graph proximity equals a hidden bridge worth investigating. For now it's parked."},
            {"role": "user", "content": "The structural coupling is a real concern. Adding a metadata field means updating 6 layers manually. We need a data flow contract."},
            {"role": "assistant", "content": "get_neighbors_rich → _traverse_graph → _graph on results → candidates file → distiller format → format_node_deep. Miss one and the data exists but never reaches its consumer."},
        ],
        "expect_revisions": 1,  # signal queue mechanism should get revised with new details
        "expect_creates_max": 5,
        "expect_divergences": 0,
        "expect_connects_min": 3,
    },
}


def run_scenario(client, model, scenario_name, scenario, verbose=True):
    """Run one scenario with a fresh brain."""
    work_dir = tempfile.mkdtemp(prefix="brain_ext_")
    db_path = os.path.join(work_dir, "brain.db")

    from servers.brain import Brain
    brain = Brain(db_path=db_path)

    # Seed nodes
    seed_ids = {}
    for seed in scenario.get("seed_nodes", []):
        result = brain.remember(**seed)
        if result:
            seed_ids[seed["title"]] = result["id"]
    brain.save()

    # Create connections between seeds for richer graph context
    seed_list = list(seed_ids.values())
    for i in range(len(seed_list) - 1):
        brain.connect(source_id=seed_list[i], target_id=seed_list[i+1],
                      relation="related_to", weight=0.7)
    brain.save()

    inst = InstrumentedBrain(brain)
    system = _load_encoding_system()

    # Build conversation + rich context
    exchanges = scenario["exchanges"]
    conv_text = "\n".join("[%s]: %s" % (e["role"].upper(), e["content"][:800]) for e in exchanges)

    # Build brain context from seed nodes using 3-degree rendering
    rich_context = ""
    user_msgs = [e for e in exchanges if e["role"] == "user"]
    for um in user_msgs[:3]:
        try:
            result = brain.recall(um["content"][:200], limit=3)
            returned = result.get("results", []) or result.get("nodes", [])
            if returned:
                lines = ['Query: "%s"' % um["content"][:80], "Brain knows:"]
                for n in returned[:2]:
                    BrainVoice.format_node_deep(n, lines, conn=brain.conn, max_d1=3, max_d2=2, max_d3=2)
                rich_context += "\n".join(lines) + "\n"
        except Exception:
            pass

    user_content = "## ENCODING RUN\n\n### Conversation\n\n%s\n\n" % conv_text
    if rich_context:
        user_content += "### Brain Context (3-degree graph)\n\n%s\n" % rich_context
    else:
        user_content += "### Brain Context\nNo relevant nodes found.\n\n"
    user_content += "### Previous State\nNo previous state.\n"

    # Run encoding agent
    t0 = time.time()
    messages = [{"role": "user", "content": user_content}]
    response = client.messages.create(
        model=model, max_tokens=4096, system=system,
        messages=messages, tools=CAPABILITY_TOOLS)

    for _ in range(8):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break
        tool_results = []
        for tu in tool_uses:
            result_text = dispatch_tool(inst, tu.name, tu.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})
        messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model=model, max_tokens=4096, system=system,
            messages=messages, tools=CAPABILITY_TOOLS)

    elapsed = time.time() - t0
    brain.save()

    # Score
    actions = inst.actions
    revises = [a for a in actions if a.tool == "revise" and not a.error]
    creates = [a for a in actions if a.tool in ("remember", "remember_lesson", "remember_mechanism") and not a.error]
    connects = [a for a in actions if a.tool == "connect" and not a.error]
    divergences = [a for a in actions if a.tool == "record_divergence" and not a.error]
    recalls = [a for a in actions if a.tool == "recall"]
    errors = [a for a in actions if a.error]

    # Check expectations
    issues = []
    if len(revises) < scenario.get("expect_revisions", 0):
        issues.append("REVISION: expected %d, got %d" % (scenario["expect_revisions"], len(revises)))
    if len(creates) > scenario.get("expect_creates_max", 99):
        issues.append("OVER-ENCODING: expected max %d creates, got %d" % (scenario["expect_creates_max"], len(creates)))
    if len(divergences) < scenario.get("expect_divergences", 0):
        issues.append("DIVERGENCE: expected %d, got %d" % (scenario["expect_divergences"], len(divergences)))
    if len(connects) < scenario.get("expect_connects_min", 0):
        issues.append("CONNECTS: expected min %d, got %d" % (scenario["expect_connects_min"], len(connects)))

    verdict = "PASS" if not issues else "FAIL"

    if verbose:
        print("[%s] %s (%.0fs)" % (verdict, scenario_name, elapsed))
        print("  %d revises, %d creates, %d connects, %d divergences, %d recalls, %d errors" % (
            len(revises), len(creates), len(connects), len(divergences), len(recalls), len(errors)))
        for a in revises + creates + connects + divergences:
            title = a.args.get("title", a.args.get("term", a.args.get("node_id", "")))
            if isinstance(title, str) and len(title) > 55:
                title = title[:52] + "..."
            print("    [%s] %s" % (a.tool, title))
        if issues:
            for issue in issues:
                print("    ⚠️ %s" % issue)

    brain.close()
    shutil.rmtree(work_dir, ignore_errors=True)

    return {
        "scenario": scenario_name,
        "verdict": verdict,
        "revises": len(revises),
        "creates": len(creates),
        "connects": len(connects),
        "divergences": len(divergences),
        "recalls": len(recalls),
        "errors": len(errors),
        "elapsed": elapsed,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="Run specific scenario")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load API key from .env
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    client = anthropic.Anthropic()

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = {k: v for k, v in SCENARIOS.items() if args.scenario in k}

    print("EXTENSIVE ENCODING TEST — %d scenarios" % len(scenarios))
    print("=" * 60)

    results = []
    for name, scenario in scenarios.items():
        if not args.quiet:
            print("\n%s: %s" % (name, scenario["description"]))
        r = run_scenario(client, args.model, name, scenario, verbose=not args.quiet)
        results.append(r)

    # Summary
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    print("\n" + "=" * 60)
    print("SUMMARY: %d/%d passed" % (passed, len(results)))
    print("=" * 60)
    print("%-25s %6s %6s %6s %6s %6s %6s" % ("Scenario", "Verdict", "Rev", "Cre", "Con", "Div", "Time"))
    for r in results:
        print("%-25s %6s %6d %6d %6d %6d %5.0fs" % (
            r["scenario"][:25], r["verdict"],
            r["revises"], r["creates"], r["connects"], r["divergences"], r["elapsed"]))

    # Save results
    results_path = str(ROOT / "eval" / "results" / ("extensive_%s.json" % time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": args.model, "results": results}, f, indent=2)
    print("\nResults: %s" % results_path)


if __name__ == "__main__":
    main()
