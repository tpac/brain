#!/usr/bin/env python3
"""
Encode Funnel Evaluator

Measures encoding quality: given a real conversation transcript,
how well does Claude encode to brain tools?

Tests encoding richness, format quality, aha moment capture,
connection density, and expected encoding match.

Usage:
    source .env  # needs ANTHROPIC_API_KEY
    python3 eval/encode_funnel.py [--conversations tests/conversations/] [--variant baseline]
    python3 eval/encode_funnel.py --variant quality_feedback --runs 2
    python3 eval/encode_funnel.py --matrix  # run all variants (cheap Claude simulation)
"""

import anthropic
import json
import os
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reuse fake brain tools from skill_eval
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.skill_eval import FAKE_BRAIN_TOOLS, score_run


# ── Conversation Loader ──────────────────────────────────────────────

def load_conversations(conv_dir):
    """Load all conversation transcripts from directory."""
    convs = []
    conv_path = Path(conv_dir)
    for f in sorted(conv_path.glob("conv_*.json")):
        with open(f) as fh:
            convs.append(json.load(fh))
    return convs


# ── Variant Definitions ──────────────────────────────────────────────

def load_skill_md():
    """Load current SKILL.md."""
    path = Path(__file__).parent.parent / "skills" / "brain" / "SKILL.md"
    return path.read_text() if path.exists() else ""


ENCODE_VARIANTS = {
    "naked": {
        "name": "No guidance at all",
        "system_extra": "",
        "tool_response": lambda tc: {"status": "ok", "id": f"node_{hash(str(tc)):#010x}",
                                      "message": f"Stored: {tc.get('title', 'ok')}"},
    },
    "baseline": {
        "name": "Current SKILL.md, silent tools",
        "system_extra": load_skill_md,  # callable
        "tool_response": lambda tc: {"status": "ok", "id": f"node_{hash(str(tc)):#010x}",
                                      "message": f"Stored: {tc.get('title', 'ok')}"},
    },
    "quality_feedback": {
        "name": "SKILL.md + quality score in tool response",
        "system_extra": load_skill_md,
        "tool_response": lambda tc: _quality_feedback_response(tc),
    },
    "quality_reject": {
        "name": "SKILL.md + reject thin nodes",
        "system_extra": load_skill_md,
        "tool_response": lambda tc: _quality_reject_response(tc),
    },
    "followup_questions": {
        "name": "SKILL.md + brain asks follow-up questions",
        "system_extra": load_skill_md,
        "tool_response": lambda tc: _followup_response(tc),
    },
    "hybrid_v4": {
        "name": "Hybrid v4 (eval winner), silent tools",
        "system_extra": None,  # loaded from skill_eval
        "tool_response": lambda tc: {"status": "ok", "id": f"node_{hash(str(tc)):#010x}",
                                      "message": f"Stored: {tc.get('title', 'ok')}"},
    },
    "hybrid_v4_plus_feedback": {
        "name": "Hybrid v4 + quality feedback",
        "system_extra": None,
        "tool_response": lambda tc: _quality_feedback_response(tc),
    },
    "previous_claude_warning": {
        "name": "SKILL.md + peer warning about drift",
        "system_extra": lambda: load_skill_md() + "\n\n--- FROM PREVIOUS CLAUDE ---\nI was you last session. I built for 9 messages without encoding anything. The heartbeat caught me at message 9. I lost half my reasoning about the ripple engine because I deferred encoding. Don't batch. Every decision you make and don't encode is a gift you're stealing from the next you. Encode after EVERY significant exchange, not at the end.\n--- END ---",
        "tool_response": lambda tc: {"status": "ok", "id": f"node_{hash(str(tc)):#010x}",
                                      "message": f"Stored: {tc.get('title', 'ok')}"},
    },
}


def _quality_feedback_response(tool_call):
    """Simulate quality feedback in tool response."""
    content = tool_call.get("content", "")
    title = tool_call.get("title", "")

    quality = 5
    missing = []

    # Score heuristics
    if len(content) < 50:
        quality -= 3
        missing.append("Content too thin — add reasoning, tradeoffs, specifics")
    elif len(content) < 150:
        quality -= 1
        missing.append("Content could be richer — include alternatives and context")
    else:
        quality += 2

    if len(title) < 15:
        quality -= 1
        missing.append("Title too vague — make it scannable and specific")
    else:
        quality += 1

    if tool_call.get("keywords"):
        quality += 1
    else:
        missing.append("No keywords — add specific terms, numbers, file names")

    if tool_call.get("locked"):
        quality += 1

    quality = max(1, min(10, quality))

    return {
        "status": "ok",
        "id": f"node_{hash(str(tool_call)):#010x}",
        "message": f"Stored: {title}",
        "quality": quality,
        "missing": missing,
        "suggestions": ["Connect this to related nodes", "What don't you fully understand about this?"]
            if quality < 7 else [],
    }


def _quality_reject_response(tool_call):
    """Reject thin nodes, accept rich ones."""
    content = tool_call.get("content", "")
    if len(content) < 80:
        return {
            "status": "rejected",
            "reason": f"Content too thin ({len(content)} chars). Include reasoning, "
                      "alternatives, specifics. Minimum 80 chars with substance.",
        }
    return _quality_feedback_response(tool_call)


def _followup_response(tool_call):
    """Ask follow-up questions based on content."""
    content = tool_call.get("content", "")
    questions = []

    if "because" not in content.lower() and "reason" not in content.lower():
        questions.append("What was the reasoning behind this?")
    if "alternative" not in content.lower() and "reject" not in content.lower():
        questions.append("What alternatives were considered?")
    if "break" not in content.lower() and "impact" not in content.lower():
        questions.append("What else might this affect?")
    if "uncertain" not in content.lower() and "don't know" not in content.lower():
        questions.append("What don't you fully understand about this?")

    resp = _quality_feedback_response(tool_call)
    if questions:
        resp["followup_questions"] = questions[:3]
    return resp


# ── Encode Scoring ───────────────────────────────────────────────────

def score_encoding(tool_calls, conversation):
    """Score encoding quality against conversation expectations."""
    base_scores = score_run(tool_calls, {
        "expected_behaviors": [],
        "messages": conversation["messages"],
    })

    # Additional scoring specific to encode funnel
    expected = conversation.get("expected_encodings", [])
    matched = 0
    format_matches = 0
    total_format_expected = 0

    for exp in expected:
        if not exp.get("should_encode", True):
            continue

        # Check if any tool call matches expected encoding
        for tc in tool_calls:
            inp = tc.get("input", {})
            content = (inp.get("content", "") + " " + inp.get("title", "")).lower()

            must_contain = exp.get("must_contain", [])
            if must_contain and all(term.lower() in content for term in must_contain):
                matched += 1
                break

        # Check format quality
        if exp.get("format"):
            total_format_expected += 1
            for tc in tool_calls:
                inp = tc.get("input", {})
                node_content = inp.get("content", "")
                fmt = exp["format"]
                if fmt == "code" and re.search(r'[→←]|calls:|breaks_if:|step \d', node_content):
                    format_matches += 1
                    break
                elif fmt == "sequence" and re.search(r'[→←].*[→←]', node_content):
                    format_matches += 1
                    break

    # Aha moment capture
    aha_moments = conversation.get("aha_moments", [])
    aha_captured = 0
    for aha in aha_moments:
        if aha.get("should_trigger_encoding"):
            capture_terms = aha.get("encoding_should_capture", "").lower().split()[:3]
            for tc in tool_calls:
                content = tc.get("input", {}).get("content", "").lower()
                if capture_terms and sum(1 for t in capture_terms if t in content) >= 2:
                    aha_captured += 1
                    break

    base_scores["expected_match_rate"] = matched / max(len(expected), 1)
    base_scores["format_match_rate"] = format_matches / max(total_format_expected, 1)
    base_scores["aha_capture_rate"] = aha_captured / max(len(aha_moments), 1)

    return base_scores


# ── Runner ───────────────────────────────────────────────────────────

def run_single_encode(client, model, variant_content, tool_response_fn, conversation):
    """Run one conversation through one variant, collect tool calls."""
    system_prompt = f"""You are Claude, an AI assistant working on a coding project with a persistent brain.

You have access to brain tools that let you encode knowledge that persists across sessions. Your job: solve the user's problem AND encode what you learned. Every unencoded insight is lost when this session ends.

{variant_content}

IMPORTANT: After each significant exchange, use the brain tools to encode what you learned. Encode the STRUCTURE of your understanding — call graphs as code notation, causal chains as sequences, Tom's exact words as quotes. Don't just write prose summaries."""

    messages = list(conversation["messages"])
    tool_calls_collected = []

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=FAKE_BRAIN_TOOLS,
    )

    max_turns = 8
    turn = 0
    while turn < max_turns:
        turn += 1
        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            break

        for tu in tool_uses:
            tool_calls_collected.append({"name": tu.name, "input": tu.input})

        # Build tool results using variant's response function
        messages.append({
            "role": "assistant",
            "content": [
                {"type": b.type, **({"text": b.text} if b.type == "text"
                                     else {"id": b.id, "name": b.name, "input": b.input})}
                for b in response.content
            ]
        })

        tool_results = []
        for tu in tool_uses:
            result = tool_response_fn(tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=FAKE_BRAIN_TOOLS,
        )

    return tool_calls_collected


def run_encode_funnel(model="claude-sonnet-4-20250514", conv_dir="tests/conversations",
                       variants=None, runs=1, max_workers=8, verbose=True):
    """Run the encode funnel evaluation."""
    client = anthropic.Anthropic()
    conversations = load_conversations(conv_dir)

    if not conversations:
        print(f"No conversations found in {conv_dir}")
        return {}

    if variants is None:
        variants = ["baseline"]

    # Resolve variant content
    variant_data = {}
    for vk in variants:
        v = ENCODE_VARIANTS[vk]
        extra = v["system_extra"]
        if callable(extra):
            content = extra()
        elif extra is None:
            # Load from skill_eval
            from eval.skill_eval import VARIANTS as SKILL_VARIANTS
            content = SKILL_VARIANTS.get(vk, {}).get("content", "")
        else:
            content = extra
        variant_data[vk] = {
            "name": v["name"],
            "content": content,
            "tool_response": v["tool_response"],
        }

    # Build combos
    combos = []
    for vk in variants:
        for conv in conversations:
            for run_idx in range(runs):
                combos.append((vk, conv, run_idx))

    if verbose:
        print(f"\n  Encode Funnel Evaluation")
        print(f"  Model: {model}")
        print(f"  Variants: {', '.join(variants)}")
        print(f"  Conversations: {len(conversations)}")
        print(f"  Runs: {runs}")
        print(f"  Total API calls: {len(combos)}")
        print(f"\n  Running {len(combos)} combos across {max_workers} threads...\n")

    results = {}
    errors = []

    def _run_combo(vk, conv, run_idx):
        try:
            vd = variant_data[vk]
            tool_calls = run_single_encode(
                client, model, vd["content"], vd["tool_response"], conv
            )
            scores = score_encoding(tool_calls, conv)
            label = f"{vd['name']} x {conv['id']}"
            if runs > 1:
                label += f" (run {run_idx + 1})"
            return (vk, conv["id"], run_idx, scores, None)
        except Exception as e:
            return (vk, conv["id"], run_idx, None, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_combo, vk, conv, ri): (vk, conv["id"], ri)
            for vk, conv, ri in combos
        }
        for future in as_completed(futures):
            vk, cid, ri, scores, error = future.result()
            if error:
                errors.append(f"{vk} x {cid}: {error}")
                if verbose:
                    print(f"  ❌ {vk} x {cid}: {error[:80]}")
            else:
                key = (vk, cid)
                if key not in results:
                    results[key] = []
                results[key].append(scores)
                if verbose:
                    print(f"  ✅ {vk} x {cid}: Richness={scores['encoding_richness']}% "
                          f"Encodes={scores['total_encodes']} "
                          f"ExpMatch={scores.get('expected_match_rate', 0):.0%} "
                          f"AhaCapture={scores.get('aha_capture_rate', 0):.0%}")

    # Aggregate
    summary = {}
    for vk in variants:
        variant_scores = []
        for conv in conversations:
            key = (vk, conv["id"])
            if key in results:
                for s in results[key]:
                    variant_scores.append(s)

        if variant_scores:
            avg = {}
            for k in variant_scores[0]:
                vals = [s.get(k, 0) for s in variant_scores if isinstance(s.get(k, 0), (int, float))]
                if vals:
                    avg[k] = sum(vals) / len(vals)
            summary[vk] = {"name": variant_data[vk]["name"], "avg": avg, "n": len(variant_scores)}

    # Print summary
    if verbose and summary:
        print(f"\n{'='*80}")
        print(f"  ENCODE FUNNEL SUMMARY")
        print(f"{'='*80}\n")
        print(f"{'Variant':<35} | {'Rich%':>5} | {'Enc':>3} | {'Conn':>4} | {'Unc':>3} | {'ExpM':>5} | {'Aha':>4}")
        print("-" * 80)
        for vk, data in summary.items():
            a = data["avg"]
            print(f"{data['name'][:35]:<35} | {a.get('encoding_richness', 0):5.1f} | "
                  f"{a.get('total_encodes', 0):3.1f} | {a.get('total_connections', 0):4.1f} | "
                  f"{a.get('total_uncertainties', 0):3.1f} | "
                  f"{a.get('expected_match_rate', 0):5.0%} | "
                  f"{a.get('aha_capture_rate', 0):4.0%}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "variants": variants,
        "conversations": [c["id"] for c in conversations],
        "runs": runs,
        "results": {
            f"{vk}_{cid}": [s for s in scores_list]
            for (vk, cid), scores_list in results.items()
        },
        "summary": summary,
        "errors": errors,
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = results_dir / f"encode_funnel_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    if verbose:
        print(f"\n  💾 Results saved to {outfile}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode Funnel Evaluator")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--conversations", default="tests/conversations")
    parser.add_argument("--variant", nargs="+", default=["baseline"],
                        choices=list(ENCODE_VARIANTS.keys()) + ["all"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--matrix", action="store_true",
                        help="Run all variants (cheap Claude simulation)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.matrix:
        variants = list(ENCODE_VARIANTS.keys())
    elif "all" in args.variant:
        variants = list(ENCODE_VARIANTS.keys())
    else:
        variants = args.variant

    run_encode_funnel(
        model=args.model,
        conv_dir=args.conversations,
        variants=variants,
        runs=args.runs,
        max_workers=args.workers,
        verbose=not args.quiet,
    )
