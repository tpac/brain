#!/usr/bin/env python3
"""Simulate the encoding agent on real conversation data with a copy of the production brain.

Feeds real exchanges in batches of 10 (like the agent runs every 5 stops),
captures all actions, and reports what the agent would do.

Usage:
    export ANTHROPIC_API_KEY=...
    python3 eval/simulate_real.py --batches 5   # Run 5 batches (50 exchanges)
    python3 eval/simulate_real.py --batches 10  # Run 10 batches (100 exchanges)
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

from eval.capabilities.base import (
    InstrumentedBrain, CAPABILITY_TOOLS, dispatch_tool, _load_encoding_system,
    CapturedAction
)
import anthropic


def extract_exchanges(transcript_path: str, limit: int = 200) -> list:
    """Extract user/assistant exchanges from a JSONL transcript."""
    exchanges = []
    with open(transcript_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                msg = entry.get("message", {})
                role = msg.get("role", entry.get("type", ""))

                if role in ("human", "user"):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
                        text = " ".join(texts)
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""
                    if text and len(text) > 5:
                        exchanges.append({"role": "user", "content": text[:1000]})

                elif role == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        texts = [p.get("text", "") for p in content
                                 if isinstance(p, dict) and p.get("type") == "text" and p.get("text")]
                        text = " ".join(texts)
                    elif isinstance(content, str):
                        text = content
                    else:
                        text = ""
                    if text and len(text) > 20:
                        exchanges.append({"role": "assistant", "content": text[:1000]})
            except Exception:
                continue

            if len(exchanges) >= limit:
                break

    return exchanges


def run_batch(client, model, brain, exchanges, batch_num, verbose=True):
    """Run encoding agent on one batch of 10 exchanges."""
    conv_text = "\n".join(
        "[%s]: %s" % (ex["role"].upper(), ex["content"][:800])
        for ex in exchanges
    )

    system = _load_encoding_system()

    messages = [
        {"role": "user",
         "content": "BATCH %d — Here are the latest 10 conversation exchanges:\n\n%s\n\n"
                    "Search the brain. Revise stale nodes. Encode what's genuinely new. Skip noise."
                    % (batch_num, conv_text)}
    ]

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=messages,
        tools=CAPABILITY_TOOLS,
    )

    for _ in range(8):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            result_text = dispatch_tool(brain, tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_text,
            })

        messages.append({
            "role": "assistant",
            "content": [
                {"type": b.type, **({"text": b.text} if b.type == "text" else
                                    {"id": b.id, "name": b.name, "input": b.input})}
                for b in response.content
            ]
        })
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=messages,
            tools=CAPABILITY_TOOLS,
        )

    brain.save()

    # Get final text response
    final_text = ""
    for b in response.content:
        if b.type == "text":
            final_text += b.text

    return final_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=5, help="Number of 10-exchange batches to run")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--transcript", help="Path to transcript JSONL")
    parser.add_argument("--start-from-end", action="store_true", help="Start from most recent exchanges")
    args = parser.parse_args()

    # Find transcript
    if args.transcript:
        transcript_path = args.transcript
    else:
        # Find the largest (current session) transcript
        transcripts = sorted(
            Path("~/.claude/projects/-Users-tpac-brain/").expanduser().glob("*.jsonl"),
            key=lambda p: p.stat().st_size, reverse=True)
        if not transcripts:
            print("No transcripts found")
            return
        transcript_path = str(transcripts[0])

    print("Transcript: %s (%.1fMB)" % (Path(transcript_path).name, Path(transcript_path).stat().st_size / 1e6))

    # Extract exchanges
    total_needed = args.batches * 10
    all_exchanges = extract_exchanges(transcript_path, limit=total_needed + 50)
    print("Extracted %d exchanges" % len(all_exchanges))

    if args.start_from_end:
        all_exchanges = all_exchanges[-total_needed:]
    else:
        all_exchanges = all_exchanges[:total_needed]

    # Copy production brain
    work_dir = tempfile.mkdtemp(prefix="brain_sim_")
    src_db = os.path.expanduser("~/AgentsContext/brain/brain.db")
    dst_db = os.path.join(work_dir, "brain.db")
    shutil.copy2(src_db, dst_db)

    # Copy logs db too
    src_logs = os.path.expanduser("~/AgentsContext/brain/brain_logs.db")
    if os.path.exists(src_logs):
        shutil.copy2(src_logs, os.path.join(work_dir, "brain_logs.db"))

    print("Brain copied to %s" % work_dir)

    from servers.brain import Brain
    brain = Brain(db_path=dst_db)
    instrumented = InstrumentedBrain(brain)

    client = anthropic.Anthropic()

    print("\n" + "=" * 70)
    print("SIMULATING ENCODING AGENT ON REAL CONVERSATIONS")
    print("=" * 70)
    print("Batches: %d, Model: %s" % (args.batches, args.model))
    print()

    all_actions = []
    batch_summaries = []

    for batch_num in range(args.batches):
        start = batch_num * 10
        end = start + 10
        batch = all_exchanges[start:end]

        if not batch:
            print("  No more exchanges")
            break

        # Show what this batch contains
        user_msgs = [e["content"][:60] for e in batch if e["role"] == "user"]
        print("── Batch %d (%d exchanges) ──" % (batch_num + 1, len(batch)))
        for um in user_msgs[:3]:
            print("  user: %s..." % um[:60])
        if len(user_msgs) > 3:
            print("  ... +%d more user messages" % (len(user_msgs) - 3))

        # Track actions before/after
        actions_before = len(instrumented.actions)

        t0 = time.time()
        final_text = run_batch(client, args.model, instrumented, batch, batch_num + 1)
        elapsed = time.time() - t0

        # Get new actions from this batch
        new_actions = instrumented.actions[actions_before:]
        all_actions.extend(new_actions)

        # Summarize
        action_counts = {}
        for a in new_actions:
            action_counts[a.tool] = action_counts.get(a.tool, 0) + 1

        errors = [a for a in new_actions if a.error]

        print("  Actions: %s (%.0fs)" % (
            ", ".join("%s:%d" % (k, v) for k, v in sorted(action_counts.items())),
            elapsed))

        # Show write actions in detail
        for a in new_actions:
            if a.tool in ("remember", "revise", "connect", "record_divergence",
                          "learn_vocabulary", "remember_lesson", "remember_mechanism"):
                title = a.args.get("title", a.args.get("term", a.args.get("node_id", "")))
                if isinstance(title, str):
                    title = title[:60]
                status = "OK" if not a.error else "ERR: %s" % a.error[:40]
                print("    [%s] %s — %s" % (a.tool, title, status))

        if errors:
            for e in errors:
                if e.tool in ("remember", "revise"):
                    print("    ERROR: %s — %s" % (e.tool, e.error[:80]))

        # Show agent's summary
        if final_text:
            summary_lines = [l for l in final_text.strip().split("\n") if l.strip() and not l.startswith("```")]
            for sl in summary_lines[:5]:
                print("  > %s" % sl[:80])

        batch_summaries.append({
            "batch": batch_num + 1,
            "actions": action_counts,
            "errors": len(errors),
            "elapsed": elapsed,
        })
        print()

    # Final summary
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    total_actions = {}
    for a in all_actions:
        total_actions[a.tool] = total_actions.get(a.tool, 0) + 1

    print("\nTotal actions across %d batches:" % len(batch_summaries))
    for tool, count in sorted(total_actions.items(), key=lambda x: -x[1]):
        print("  %-25s %d" % (tool, count))

    writes = [a for a in all_actions if a.tool in ("remember", "revise", "connect",
              "record_divergence", "learn_vocabulary", "remember_lesson", "remember_mechanism")]
    revises = [a for a in all_actions if a.tool == "revise"]
    creates = [a for a in all_actions if a.tool in ("remember", "remember_lesson", "remember_mechanism")]
    errors = [a for a in all_actions if a.error]

    print("\nWrite breakdown:")
    print("  Revisions: %d" % len(revises))
    print("  New nodes: %d" % len(creates))
    print("  Connects:  %d" % len([a for a in all_actions if a.tool == "connect"]))
    print("  Errors:    %d" % len(errors))
    print("  Revision ratio: %.0f%%" % (len(revises) / max(1, len(revises) + len(creates)) * 100))

    # Save results
    results_path = os.path.join(str(ROOT / "eval" / "results"),
                                "simulation_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "batches": len(batch_summaries),
            "total_actions": total_actions,
            "revisions": len(revises),
            "new_nodes": len(creates),
            "errors": len(errors),
            "batch_summaries": batch_summaries,
            "actions_detail": [
                {"tool": a.tool, "args_summary": {k: str(v)[:100] for k, v in a.args.items()},
                 "error": a.error}
                for a in writes
            ],
        }, f, indent=2)
    print("\nResults: %s" % results_path)

    # Cleanup
    brain.close()
    shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
