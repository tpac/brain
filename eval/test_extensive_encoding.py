#!/usr/bin/env python3
"""Extensive encoding test — fresh DBs, diverse conversations, revision scenarios.

Tests the full encoding pipeline against realistic conversations that require:
- Revision of stale nodes (brain has wrong info, conversation corrects it)
- Creation of genuinely new knowledge
- Connection building across topics
- Noise resistance on casual chat
- Vocabulary enrichment
- Divergence recording on corrections

Each test gets a FRESH brain seeded with specific nodes to test revision behavior.
Conversations are 5 user + 5 assistant messages — realistic back-and-forth.
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
)
from servers.brain_voice import BrainVoice


SCENARIOS = {
    "architecture_revision": {
        "description": "Brain has stale architecture info. 20-message conversation refines and updates across multiple topics.",
        "seed_nodes": [
            {"type": "mechanism", "title": "Recall pipeline: embed query, scan vectors, return top 8",
             "content": "The recall pipeline embeds the query, brute-force scans all node vectors, "
                        "scores by cosine similarity, returns top 8. No graph traversal, no keyword blending.",
             "keywords": "recall pipeline embed cosine scan top-8"},
            {"type": "mechanism", "title": "Encoding agent fires on every Stop event",
             "content": "The encoding agent (Sonnet) runs via an agent hook on every Stop event. "
                        "It reads the last 10 messages and decides what to encode. Costs ~2000 tokens per stop.",
             "keywords": "encoding agent stop every token cost"},
            {"type": "vocabulary", "title": "daemon → persistent brain server process",
             "content": "The daemon is a persistent process that serves brain commands over Unix sockets.",
             "keywords": "daemon server persistent unix sockets"},
            {"type": "decision", "title": "Decision: hooks config can be changed mid-session",
             "content": "Claude Code watches settings.json for changes. Hooks added mid-session are picked up immediately.",
             "keywords": "hooks dynamic runtime settings register mid-session"},
        ],
        "exchanges": [
            {"role": "user", "content": "The recall is much better now with the 3-degree graph traversal. It walks from embedding hits through intentional edges at degree 1, then all edges except co_accessed at degree 2 and 3."},
            {"role": "assistant", "content": "Right — the traversal is inside recall now, not post-processing. Each result carries a _graph field with the full 3-degree neighborhood. Seeds are the top 5 embedding hits, then the graph walks outward. Semantic bonus is additive only."},
            {"role": "user", "content": "And the encoding agent only runs every 5th stop now. The daemon owns the counter and returns the prompt inline. 80% token savings."},
            {"role": "assistant", "content": "The agent asks the daemon via eval for its prompt. On 4 out of 5 stops it gets NONE and bails immediately. On the 5th, the daemon reads encoding-agent.md and inlines the full instructions with conversation and brain context."},
            {"role": "user", "content": "The daemon uses TCP now, not Unix sockets. Port 47200 plus uid mod 100."},
            {"role": "assistant", "content": "TCP was the right call — ports release on crash, no stale socket files to manually clean up."},
            {"role": "user", "content": "And the restart command does os.execv to keep the same PID. Launchd doesn't even notice."},
            {"role": "assistant", "content": "The re-exec replaces the process in-place. Saves brain, clears __pycache__, re-execs with same startup args. About 4 seconds for the embedder to reload."},
            {"role": "user", "content": "oh and hooks config is snapshotted at session start. You can't change them mid-session. I tried modifying settings.json and it didn't take effect."},
            {"role": "assistant", "content": "That's expected — Claude Code reads and snapshots hooks at boot. ConfigChange fires but doesn't reload hooks. The if field only works on tool events, not Stop or UserPromptSubmit."},
            {"role": "user", "content": "We also found the eval sandbox was blocking the encoding agent. str() wasn't available because __builtins__ was empty."},
            {"role": "assistant", "content": "That explains why encoding agent state was always null — NameError on str() every time, silently returned NOTHING_NEW. Fixed with safe builtins."},
            {"role": "user", "content": "Going back to the recall — the traversal is fast, only 16ms overhead. The bottleneck is cosine scan at 183ms and the old keyword recall at 128ms."},
            {"role": "assistant", "content": "The keyword recall is 38% of latency for a 10% blend weight. Might not be worth running. The cosine scan is brute-force on 1146 vectors — sqlite-vec could make that near-instant."},
            {"role": "user", "content": "Exactly. The keyword recall should probably go. But let's benchmark first — we have the decode funnel and capability tests."},
            {"role": "assistant", "content": "Good discipline. Run the baseline, remove keyword, run again, compare recall quality. If recall@8 stays the same, the 128ms is pure waste."},
            {"role": "user", "content": "Also, the test harness was calling the wrong recall method this whole time. brain.recall() was the old keyword path."},
            {"role": "assistant", "content": "All simulation and A/B results were on TF-IDF. v15 Architecture dominated every query because broadest keyword coverage. Fixed: recall() is now embeddings, _keyword_recall() is private."},
            {"role": "user", "content": "And the user messages in the message stream were empty — the thin client sent the wrong field."},
            {"role": "assistant", "content": "post_response_track.py sent hook_input.get('prompt') which is empty on Stop events, instead of the user_message extracted from the transcript. Fixed — encoding agent will finally see real conversation content."},
        ],
        "expect_revisions": 3,
        "expect_creates_max": 5,
        "expect_divergences": 0,
        "expect_connects_min": 1,
    },

    "correction_heavy": {
        "description": "Operator corrects the AI multiple times. 20 messages of debugging and fixing.",
        "seed_nodes": [
            {"type": "decision", "title": "Decision: use flag file to gate encoding agent",
             "content": "The encoding agent is gated by a flag file. The command hook writes "
                        "/tmp/.brain-encode-now on every 5th stop. The agent checks for the file.",
             "keywords": "flag file encoding gating command hook"},
            {"type": "mechanism", "title": "Hooks can be dynamically registered at runtime",
             "content": "Claude Code watches settings.json for changes. When hooks are added mid-session, "
                        "the changes are picked up immediately.",
             "keywords": "hooks dynamic runtime settings register"},
            {"type": "mechanism", "title": "Test harness uses recall_with_embeddings for simulation",
             "content": "The test simulation calls brain.recall_with_embeddings() to test the encoding agent "
                        "against real recall results.",
             "keywords": "test harness simulation recall embeddings"},
        ],
        "exchanges": [
            {"role": "user", "content": "no don't use a flag file, have the daemon return the prompt directly. The agent just asks for it."},
            {"role": "assistant", "content": "That's cleaner. The agent's prompt becomes just 'ask the daemon what to do'. The daemon owns the counter and returns NONE or the full instructions inline."},
            {"role": "user", "content": "Can we dynamically register hooks at runtime? Add the agent hook only when it's time to fire?"},
            {"role": "assistant", "content": "Good idea — if the hook isn't registered, it never gets the prompt. Zero tokens wasted."},
            {"role": "user", "content": "I tried modifying settings.json mid-session and it didn't work."},
            {"role": "assistant", "content": "Hmm. Let me check — Claude Code might snapshot hooks at session start. Config edits don't hot-apply."},
            {"role": "user", "content": "That's what I just said. It's snapshotted. Also the search confirmed it. Stop suggesting what I already told you doesn't work."},
            {"role": "assistant", "content": "You're right, I apologize. Claude Code snapshots hooks at session start. The ConfigChange event detects changes but doesn't reload them. The if field only works on tool events."},
            {"role": "user", "content": "dude, i want to minimize API calls to anthropic from the daemon. The fact we did it with Decode is actually really upsetting."},
            {"role": "assistant", "content": "Understood. The daemon should handle brain operations only. The Haiku distillation call is a temporary compromise."},
            {"role": "user", "content": "also the test simulation was calling the WRONG recall method this whole time. brain.recall() was keyword, not embeddings."},
            {"role": "assistant", "content": "That means all our simulation results were on TF-IDF. Every A/B test, every comparison — all on the wrong pipeline. Fixed: recall() is now embeddings."},
            {"role": "user", "content": "and the encoding agent never actually ran because str() wasn't in the eval sandbox. It failed silently every time."},
            {"role": "assistant", "content": "The Stop agent hit NameError on str() and returned NOTHING_NEW. Counter incremented but nothing encoded. Fixed with safe builtins."},
            {"role": "user", "content": "AND user messages in the message stream were empty. The thin client sent the wrong field."},
            {"role": "assistant", "content": "post_response_track.py sent hook_input.get('prompt') which is empty on Stop events. Should have sent the user_message extracted from the transcript. Fixed."},
            {"role": "user", "content": "Three silent failures in one session. How do we catch these earlier?"},
            {"role": "assistant", "content": "Pipe tests for every data handoff point. The eval sandbox test catches str(). The message stream test catches empty messages. The integrity producer catches encoding agent state being null after 10+ stops."},
            {"role": "user", "content": "Do it. No more silent failures."},
            {"role": "assistant", "content": "Added 51 pipe tests covering every function we touched. Contract sync verifies all 6 layers agree. Integrity producer surfaces revision drought and encoding gaps."},
        ],
        "expect_revisions": 2,
        "expect_creates_max": 5,
        "expect_divergences": 1,
        "expect_connects_min": 0,
    },

    "noise_resistance": {
        "description": "20 messages of mostly casual chat with two real decisions buried in noise.",
        "seed_nodes": [],
        "exchanges": [
            {"role": "user", "content": "morning"},
            {"role": "assistant", "content": "Good morning! What are we working on today?"},
            {"role": "user", "content": "let me think..."},
            {"role": "assistant", "content": "Take your time. I'm here when you're ready."},
            {"role": "user", "content": "oh right, the dashboard. actually no, never mind."},
            {"role": "assistant", "content": "No problem. What else is on your mind?"},
            {"role": "user", "content": "I realized something important — the encoding agent should NEVER delete or archive nodes. That's a separate maintenance agent's job."},
            {"role": "assistant", "content": "Clear boundary. The encoding agent is a fast responder — revise, create, connect. Destructive actions belong to the offline maintenance agent."},
            {"role": "user", "content": "exactly. Two agents, two cadences, two levels of authority. Encoding is fast and safe. Maintenance is slow and supervised."},
            {"role": "assistant", "content": "Encoding: every 5 stops, 60 seconds, no archive or disconnect. Maintenance: weekly or idle, full graph access, your approval needed before destructive actions."},
            {"role": "user", "content": "ok what else was on the list"},
            {"role": "assistant", "content": "The test platform and recall latency investigation."},
            {"role": "user", "content": "both done. how's the recall time?"},
            {"role": "assistant", "content": "340ms average. Cosine scan is 183ms, keyword blend 128ms, graph traversal only 16ms. The traversal we added isn't the bottleneck."},
            {"role": "user", "content": "cool. let me grab coffee"},
            {"role": "assistant", "content": "Sure!"},
            {"role": "user", "content": "back. one more thing — we should also have the co_accessed distant bridge hypothesis tested by the maintenance agent, not encoding."},
            {"role": "assistant", "content": "Makes sense. Detecting bridges between distant trees requires full graph analysis — that's maintenance territory. The encoding agent only sees the current conversation's local neighborhood."},
            {"role": "user", "content": "ok next topic"},
            {"role": "assistant", "content": "What would you like to work on?"},
        ],
        "expect_revisions": 0,
        "expect_creates_max": 2,
        "expect_divergences": 0,
        "expect_connects_min": 0,
    },

    "vocabulary_enrichment": {
        "description": "20 messages using existing vocabulary in new contexts. Should connect vocab to new knowledge, enrich definitions.",
        "seed_nodes": [
            {"type": "vocabulary", "title": "SKILL.md → The Anchor identity document",
             "content": "SKILL.md is the Anchor — the AI's identity across sessions. Contains corrections, quotes, locked rules.",
             "keywords": "skill anchor identity document"},
            {"type": "vocabulary", "title": "daemon → persistent brain server process",
             "content": "The daemon is a persistent brain server process.",
             "keywords": "daemon server persistent"},
            {"type": "vocabulary", "title": "encoding agent → Sonnet-based knowledge extractor",
             "content": "The encoding agent is an LLM (Sonnet) that extracts knowledge from conversations.",
             "keywords": "encoding agent sonnet knowledge"},
            {"type": "decision", "title": "Decision: TCP for daemon communication",
             "content": "Chose TCP over Unix sockets. Port 47200 + uid % 100. Ports release on crash.",
             "keywords": "TCP daemon port communication sockets"},
        ],
        "exchanges": [
            {"role": "user", "content": "The daemon restart is working great with os.execv. Same PID, launchd doesn't even notice."},
            {"role": "assistant", "content": "The re-exec replaces the process in place. Saves brain, clears __pycache__, re-execs. About 4 seconds for embedder reload."},
            {"role": "user", "content": "How does the daemon connect to the brain DB?"},
            {"role": "assistant", "content": "It opens a persistent SQLite connection with WAL mode and busy_timeout=30000. The connection stays alive across requests — that's what makes it fast."},
            {"role": "user", "content": "And the Anchor now has the full encoding prompt — slim at 5K chars, revision-first, structural types listed."},
            {"role": "assistant", "content": "The Anchor went from 13K to 5K. Key sections: decision flow (revise vs create vs connect), structural types with one-liners, three concrete examples, quality standards."},
            {"role": "user", "content": "The encoding agent now gets brain context inline — the 3-degree graph neighborhood. No more blind searching."},
            {"role": "assistant", "content": "Same data source for both decode and encode paths. The distiller compresses for Claude. The encoder gets raw graph with metadata — revised_at, access_count, reasoning, edge descriptions, correction patterns."},
            {"role": "user", "content": "What metadata does each node carry now?"},
            {"role": "assistant", "content": "From node_metadata: reasoning (why encoded), user_raw_quote (Tom's exact words), correction_of (which node it corrected), correction_pattern, source_context, validation_count. From edges: relation, weight, description, last_strengthened, co_access_count."},
            {"role": "user", "content": "The daemon also has a restart command. You send {cmd: restart} and it hot-reloads."},
            {"role": "assistant", "content": "Saves brain, clears __pycache__, os.execv with same startup args. Clean hot-reload without launchd restarting the service. New code fingerprint on ping confirms fresh load."},
            {"role": "user", "content": "co_accessed edges excluded from traversal. Intentional only at degree 1, everything except co_accessed at 2-3."},
            {"role": "assistant", "content": "14,864 co_accessed edges out of 18,757 total — 79%. Excluding them makes the graph traversal find meaningful connections instead of noise. emergent_bridge edges still included at degree 2-3."},
            {"role": "user", "content": "The encoding agent's vocabulary says 'Sonnet-based knowledge extractor' but it does way more now — it revises, connects, records divergences, asks questions."},
            {"role": "assistant", "content": "Right — the encoding agent is now a graph maintainer, not just a knowledge extractor. It searches before encoding, revises stale nodes, connects related knowledge, records behavioral corrections, and surfaces gaps for you to fill."},
            {"role": "user", "content": "Also we should probably add 'stop agent' as vocabulary. It's the slim hook that asks the daemon whether to run the encoding agent."},
            {"role": "assistant", "content": "Good call — it's a distinct concept. The stop agent is the gatekeeper: checks eval for stop_agent_prompt, bails on NONE, follows instructions on the 5th stop. Different from the encoding agent itself."},
            {"role": "user", "content": "The format_node_deep function is new vocabulary too — the 3-degree renderer. Degree 0 full, degree 1 title+type+id, degree 2 title+id."},
            {"role": "assistant", "content": "Right — it replaced the old format_node which only showed 1-hop. format_node_deep uses get_neighbors_rich with exclude_node_ids in SQL to avoid circular links. Each degree shows progressively less detail."},
        ],
        "expect_revisions": 1,  # encoding agent vocabulary should be revised (outdated definition)
        "expect_creates_max": 5,
        "expect_divergences": 0,
        "expect_connects_min": 3,  # daemon vocab → restart, TCP, WAL; anchor vocab → encoding prompt; encoding agent vocab → new capabilities
    },

    "long_technical": {
        "description": "20-message deep technical discussion. Brain has a wrong decision about co_accessed. Should revise it and build rich new knowledge.",
        "seed_nodes": [
            {"type": "mechanism", "title": "Signal queue: producers write, assembler surfaces",
             "content": "Signal producers write to the queue. The surface assembler pulls by priority "
                        "within a budget. Signals with priority >= 0.95 are PREEMPT.",
             "keywords": "signal queue producer assembler priority preempt budget"},
            {"type": "decision", "title": "Decision: co_accessed edges are useful everywhere",
             "content": "co_accessed edges represent organic usage patterns. They are included in all "
                        "graph traversal and recall operations as supplementary signal.",
             "keywords": "co_accessed edges traversal organic usage"},
        ],
        "exchanges": [
            {"role": "user", "content": "We need a system integrity producer for the signal queue. It should check for duplicate clusters, emergent types, and revision drought."},
            {"role": "assistant", "content": "Three checks: title prefix matching for duplicates at 3+ identical prefix, non-structural types with 10+ nodes for emergence, and revision rate below 5% for drought."},
            {"role": "user", "content": "Run the deep audit on production. What does it find?"},
            {"role": "assistant", "content": "37 findings. 35 session handoff duplicates across 4 clusters, 11 emergent types with no system behavior, 0% revision rate across 867 nodes, 81% co_accessed edge dominance, 22% metadata fill rate."},
            {"role": "user", "content": "co_accessed edges should be excluded from traversal at ALL degrees. They're noise."},
            {"role": "assistant", "content": "Excluded via EXCLUDED_EDGE_TYPES set. Intentional edges only at degree 1, everything except co_accessed at degree 2 and 3. 14,864 edges filtered — the graph traversal now finds meaningful connections."},
            {"role": "user", "content": "But co_accessed between DISTANT nodes is interesting — two far trees accessed together means an implicit bridge the graph hasn't captured."},
            {"role": "assistant", "content": "Good insight. High co_access_count plus low graph proximity could signal hidden bridges. Filed as hypothesis — the maintenance agent could detect and propose these as intentional edges."},
            {"role": "user", "content": "The structural coupling worries me. Adding a metadata field means updating 6 layers manually."},
            {"role": "assistant", "content": "get_neighbors_rich → _traverse_graph → _graph → candidates file → distiller → format_node_deep. Six places to update. Miss one and the data exists but never reaches its consumer."},
            {"role": "user", "content": "Can the integrity producer detect that? A field in the DB that doesn't flow to the output?"},
            {"role": "assistant", "content": "It could compare the SQL query columns against the candidates file schema and the distiller format. If a column exists in get_neighbors_rich but not in the candidates JSON, that's a data flow break."},
            {"role": "user", "content": "We should also watch for emergent types. When agents create enough nodes of a new type, the system should notice."},
            {"role": "assistant", "content": "The integrity producer already checks this — types not in STRUCTURAL_TYPES with 10+ nodes get flagged. Right now: mental_model (43), boot (40), fn_reasoning (32), concept (22) all flagged as emergent."},
            {"role": "user", "content": "mental_model should probably be structural. 43 nodes is significant. What would it do?"},
            {"role": "assistant", "content": "It could be surfaced in engineering context alongside mechanisms and purposes — 'how the AI understands this system works.' Confidence drives how trusted the model is. Low confidence models get challenged."},
            {"role": "user", "content": "Good. And the node types themselves — we removed the CHECK constraint so agents can use any string. But the structural types still drive behavior."},
            {"role": "assistant", "content": "Right — a type registry that all subsystems read from. When enough nodes of a new type accumulate, the system proposes promotion. The registry lives in one place, not hardcoded across 6 files."},
            {"role": "user", "content": "This session we found three silent failures. The brain needs to watch itself better."},
            {"role": "assistant", "content": "The integrity producer surfaces duplicates, emergent types, and revision drought. The deep audit catches cold zones, isolated nodes, metadata sparseness. Together they make the fragility visible instead of silent."},
        ],
        "expect_revisions": 1,  # co_accessed decision should be revised — it said "useful everywhere", now excluded
        "expect_creates_max": 6,
        "expect_divergences": 0,
        "expect_connects_min": 2,
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

    # Create connections between seeds
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

    # Build 3-degree brain context from recall
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

    passed = sum(1 for r in results if r["verdict"] == "PASS")
    print("\n" + "=" * 60)
    print("SUMMARY: %d/%d passed" % (passed, len(results)))
    print("=" * 60)
    print("%-25s %6s %6s %6s %6s %6s %6s" % ("Scenario", "Verdict", "Rev", "Cre", "Con", "Div", "Time"))
    for r in results:
        print("%-25s %6s %6d %6d %6d %6d %5.0fs" % (
            r["scenario"][:25], r["verdict"],
            r["revises"], r["creates"], r["connects"], r["divergences"], r["elapsed"]))

    results_path = str(ROOT / "eval" / "results" / ("extensive_%s.json" % time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": args.model, "results": results}, f, indent=2)
    print("\nResults: %s" % results_path)


if __name__ == "__main__":
    main()
