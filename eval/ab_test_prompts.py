#!/usr/bin/env python3
"""A/B test encoding agent prompt variants on real conversation data.

Runs each variant on the same batches and compares KPIs:
- Revisions vs creates
- Questions asked
- Noise resistance (NOTHING_NEW on empty batches)
- Search effort (recall calls)
- Total tool calls
- Token efficiency (actions per tool call)

Usage:
    export ANTHROPIC_API_KEY=...
    python3 eval/ab_test_prompts.py --batches 10
    python3 eval/ab_test_prompts.py --batches 20 --start-from-end
"""
import sys
import os
import json
import time
import shutil
import tempfile
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.capabilities.base import InstrumentedBrain, CAPABILITY_TOOLS, dispatch_tool
import anthropic

VARIANTS = {
    "A_current": str(ROOT / "eval" / "prompts" / "variant_a_current.md"),
    "B_curious": str(ROOT / "eval" / "prompts" / "variant_b_curious.md"),
    "C_loose":   str(ROOT / "eval" / "prompts" / "variant_c_loose.md"),
}


def extract_exchanges(transcript_path, limit=300):
    """Extract user/assistant exchanges from JSONL transcript."""
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
                        exchanges.append({"role": "user", "content": text[:800]})
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
                        exchanges.append({"role": "assistant", "content": text[:800]})
            except Exception:
                continue
            if len(exchanges) >= limit:
                break
    return exchanges


def run_variant_batch(client, model, system_prompt, brain, batch, batch_num):
    """Run one batch with one variant. Returns actions list."""
    conv_text = "\n".join("[%s]: %s" % (e["role"].upper(), e["content"][:600]) for e in batch)

    messages = [{"role": "user",
                 "content": "## ENCODING RUN\n\n### Conversation\n\n%s\n\n"
                            "### Brain Context\nNo recall data available.\n\n"
                            "### Previous State\nNo previous state.\n" % conv_text}]

    response = client.messages.create(
        model=model, max_tokens=4096, system=system_prompt,
        messages=messages, tools=CAPABILITY_TOOLS)

    for _ in range(8):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            result_text = dispatch_tool(brain, tu.name, tu.input)
            tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": result_text})

        messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model, max_tokens=4096, system=system_prompt,
            messages=messages, tools=CAPABILITY_TOOLS)

    # Extract final text
    final = ""
    for b in response.content:
        if b.type == "text":
            final += b.text

    brain.save()
    return final


def score_variant(actions, final_texts):
    """Score a variant's performance across all batches."""
    total_recalls = sum(1 for a in actions if a.tool == "recall")
    total_finds = sum(1 for a in actions if a.tool == "find_node_by_title")
    total_gets = sum(1 for a in actions if a.tool == "get_node")
    total_revises = sum(1 for a in actions if a.tool == "revise")
    total_creates = sum(1 for a in actions if a.tool in ("remember", "remember_lesson", "remember_mechanism"))
    total_connects = sum(1 for a in actions if a.tool == "connect")
    total_divergences = sum(1 for a in actions if a.tool == "record_divergence")
    total_vocab = sum(1 for a in actions if a.tool == "learn_vocabulary")
    total_errors = sum(1 for a in actions if a.error)

    # Count questions from final texts
    questions = 0
    nothing_new = 0
    for ft in final_texts:
        if "ASK_USER" in ft and "NONE" not in ft.split("ASK_USER")[1][:50]:
            questions += 1
        if "NOTHING_NEW" in ft:
            nothing_new += 1

    writes = total_revises + total_creates
    searches = total_recalls + total_finds + total_gets

    return {
        "searches": searches,
        "recalls": total_recalls,
        "finds": total_finds,
        "gets": total_gets,
        "revises": total_revises,
        "creates": total_creates,
        "connects": total_connects,
        "divergences": total_divergences,
        "vocabulary": total_vocab,
        "errors": total_errors,
        "questions_asked": questions,
        "nothing_new_count": nothing_new,
        "revision_ratio": total_revises / max(1, writes),
        "actions_per_search": writes / max(1, searches),
        "total_actions": len(actions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--start-from-end", action="store_true")
    parser.add_argument("--transcript", help="Specific transcript path")
    args = parser.parse_args()

    # Find transcript
    if args.transcript:
        transcript_path = args.transcript
    else:
        transcripts = sorted(
            Path("~/.claude/projects/-Users-tpac-brain/").expanduser().glob("*.jsonl"),
            key=lambda p: p.stat().st_size, reverse=True)
        transcript_path = str(transcripts[0])

    print("Transcript: %s" % Path(transcript_path).name)

    total_needed = args.batches * 10
    all_exchanges = extract_exchanges(transcript_path, limit=total_needed + 50)

    if args.start_from_end:
        all_exchanges = all_exchanges[-total_needed:]
    else:
        all_exchanges = all_exchanges[:total_needed]

    batches = []
    for i in range(0, len(all_exchanges), 10):
        batch = all_exchanges[i:i+10]
        if batch:
            batches.append(batch)

    print("Batches: %d, Exchanges: %d" % (len(batches), len(all_exchanges)))

    client = anthropic.Anthropic()
    results = {}

    for variant_name, prompt_path in VARIANTS.items():
        print("\n" + "=" * 60)
        print("VARIANT: %s" % variant_name)
        print("=" * 60)

        with open(prompt_path) as f:
            system_prompt = f.read()

        print("Prompt: %d chars" % len(system_prompt))

        # Fresh brain copy for each variant
        work_dir = tempfile.mkdtemp(prefix="brain_ab_")
        shutil.copy2(os.path.expanduser("~/AgentsContext/brain/brain.db"),
                      os.path.join(work_dir, "brain.db"))
        src_logs = os.path.expanduser("~/AgentsContext/brain/brain_logs.db")
        if os.path.exists(src_logs):
            shutil.copy2(src_logs, os.path.join(work_dir, "brain_logs.db"))

        from servers.brain import Brain
        brain = Brain(db_path=os.path.join(work_dir, "brain.db"))
        instrumented = InstrumentedBrain(brain)

        final_texts = []
        t0 = time.time()

        for i, batch in enumerate(batches):
            user_msgs = [e["content"][:40] for e in batch if e["role"] == "user"]
            has_user = len(user_msgs) > 0

            actions_before = len(instrumented.actions)
            final = run_variant_batch(client, args.model, system_prompt, instrumented, batch, i+1)
            new_actions = instrumented.actions[actions_before:]
            final_texts.append(final)

            action_counts = {}
            for a in new_actions:
                action_counts[a.tool] = action_counts.get(a.tool, 0) + 1

            marker = "👤" if has_user else "🤖"
            summary = ", ".join("%s:%d" % (k, v) for k, v in sorted(action_counts.items()))
            print("  %s Batch %d: %s" % (marker, i+1, summary or "no actions"))

        elapsed = time.time() - t0
        scores = score_variant(instrumented.actions, final_texts)
        scores["elapsed"] = elapsed
        scores["prompt_chars"] = len(system_prompt)
        results[variant_name] = scores

        brain.close()
        shutil.rmtree(work_dir, ignore_errors=True)

    # Comparison table
    print("\n" + "=" * 70)
    print("A/B TEST RESULTS")
    print("=" * 70)

    metrics = ["prompt_chars", "searches", "revises", "creates", "connects",
               "divergences", "vocabulary", "questions_asked", "nothing_new_count",
               "errors", "revision_ratio", "total_actions", "elapsed"]

    header = "%-22s" % "Metric"
    for vn in VARIANTS:
        header += " %12s" % vn[:12]
    print(header)
    print("-" * (22 + 13 * len(VARIANTS)))

    for m in metrics:
        row = "%-22s" % m
        for vn in VARIANTS:
            val = results[vn].get(m, 0)
            if isinstance(val, float):
                row += " %12.2f" % val
            else:
                row += " %12s" % str(val)
        print(row)

    # Save
    results_path = str(ROOT / "eval" / "results" / ("ab_test_%s.json" % time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "batches": len(batches), "model": args.model,
                    "results": results}, f, indent=2)
    print("\nResults: %s" % results_path)


if __name__ == "__main__":
    main()
