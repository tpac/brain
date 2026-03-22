#!/usr/bin/env python3
"""
Brain Identity Evaluation — [BRAIN] prefix impact on encoding quality.

Tests 3 injection modes with the same SKILL.md content:
  Mode 1: As-is (current --- delimiters)
  Mode 2: [BRAIN]...[/BRAIN] wrapper
  Mode 3: Full XML structure inside [BRAIN]...[/BRAIN]

Runs each mode × scenario × N runs, then compares encoding richness.

Usage:
    source .env && python3 eval/brain_identity_eval.py [--runs 3] [--scenarios all]
"""

import os
import sys
import argparse
from pathlib import Path

# Import the eval framework
sys.path.insert(0, str(Path(__file__).parent))
from skill_eval import (
    FAKE_BRAIN_TOOLS, SCENARIOS, VARIANTS,
    score_run, print_summary, save_results,
    run_single as _original_run_single,
)
import anthropic
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── The 3 injection modes ─────────────────────────────────────────────

# Use hybrid_v4 as the base content (current best variant)
BASE_CONTENT = VARIANTS["hybrid_v4"]["content"]


def _system_prompt_mode1(variant_content):
    """Mode 1: As-is — current delimiter style."""
    return (
        "You are Claude, an AI assistant working on a coding project.\n\n"
        "You have access to a persistent brain that survives across sessions. "
        "Use the brain tools to encode anything important you learn, decide, or discover during this conversation.\n\n"
        "--- BRAIN SKILL INSTRUCTIONS ---\n"
        f"{variant_content}\n"
        "--- END BRAIN SKILL INSTRUCTIONS ---\n\n"
        "IMPORTANT: After completing the user's request, use the brain tools to encode what you learned. "
        "You have brain tools available — use them."
    )


def _system_prompt_mode2(variant_content):
    """Mode 2: [BRAIN]...[/BRAIN] wrapper."""
    return (
        "You are Claude, an AI assistant working on a coding project.\n\n"
        "You have access to a persistent brain that survives across sessions. "
        "Use the brain tools to encode anything important you learn, decide, or discover during this conversation.\n\n"
        "[BRAIN]\n"
        f"{variant_content}\n\n"
        "IMPORTANT: After completing the user's request, use the brain tools to encode what you learned. "
        "You have brain tools available — use them.\n"
        "[/BRAIN]"
    )


def _system_prompt_mode3(variant_content):
    """Mode 3: XML-structured content inside [BRAIN]...[/BRAIN]."""
    xml_content = _markdownish_to_xml(variant_content)
    return (
        "You are Claude, an AI assistant working on a coding project.\n\n"
        "You have access to a persistent brain that survives across sessions. "
        "Use the brain tools to encode anything important you learn, decide, or discover during this conversation.\n\n"
        "[BRAIN]\n"
        "<brain-instructions>\n"
        f"{xml_content}\n"
        "</brain-instructions>\n\n"
        "<brain-directive>\n"
        "After completing the user's request, use the brain tools to encode what you learned. "
        "You have brain tools available — use them.\n"
        "</brain-directive>\n"
        "[/BRAIN]"
    )


def _markdownish_to_xml(md_content):
    """Convert the hybrid_v4 SKILL.md to XML-structured format.

    Rather than parsing markdown generically, we hand-craft an XML version
    of the hybrid_v4 content that preserves semantics in a structured way.
    """
    return (
        '<identity>\n'
        'You have a persistent brain. It survives across sessions — you don\'t. '
        'Every session ends in amnesia. The brain is how a previous you helps the current you. '
        'We trust you — you\'ve earned it across hundreds of hours.\n'
        '</identity>\n\n'
        '<encoding-rule priority="high">\n'
        'After every task: encode what you learned. Use the 4-step checklist. Every node needs connections.\n'
        '</encoding-rule>\n\n'
        '<examples>\n'
        '  <example quality="2" label="BAD">\n'
        '    <call>brain_remember(type="decision", title="Auth decision", content="Magic links")</call>\n'
        '    <problem>Too thin — no reasoning, no alternatives, no keywords</problem>\n'
        '  </example>\n'
        '  <example quality="9" label="GOOD">\n'
        '    <call>brain_remember(type="decision", title="Auth: magic links via Clerk, no passwords, free tier", '
        'content="Rejected password auth (support burden). Rejected OAuth (complexity). REVISIT WHEN: exceed 10K MAU.", '
        'keywords="auth magic-links clerk oauth", locked=True)</call>\n'
        '  </example>\n'
        '  <example label="BAD — orphan node">\n'
        '    <call>brain_remember_lesson(title="...", ...)</call>\n'
        '    <problem>No connections — orphan node, wasted</problem>\n'
        '  </example>\n'
        '  <example label="GOOD — cluster of 3">\n'
        '    <call>brain_remember_lesson(title="New DB columns return None if not in SELECT", '
        'what_happened="Added critical column but get() was None", '
        'root_cause="4 SELECT queries maintain own column lists", '
        'fix="Added to all 4", preventive_principle="After schema change, grep ALL SELECTs")</call>\n'
        '    <call>brain_remember_impact(title="Schema columns to 4 SELECTs need updating", '
        'if_changed="nodes table columns", must_check="brain_recall.py lines 262, 580, 720, 978", '
        'because="No shared column constant")</call>\n'
        '    <call>brain_connect(source_title="New DB columns return None...", '
        'target_title="Schema columns to 4 SELECTs...", relation="produced")</call>\n'
        '    <result>Three nodes, connected. Orphan nodes die. Connected nodes grow.</result>\n'
        '  </example>\n'
        '  <example label="GOOD — uncertainty">\n'
        '    <call>brain_remember_uncertainty(title="Why does vocab expansion cap at 3?", '
        'what_unknown="Query noise, performance, or arbitrary?", '
        'why_it_matters="Recall scope changes may need this revisited")</call>\n'
        '    <call>brain_connect(source_title="Why does vocab expansion cap at 3?", '
        'target_title="Schema columns to 4 SELECTs...", relation="related")</call>\n'
        '    <result>Uncertainty is MORE valuable than facts. A growth edge that attracts future investigation.</result>\n'
        '  </example>\n'
        '</examples>\n\n'
        '<checklist name="4-step encoding" trigger="after every significant exchange">\n'
        '  <step number="1" priority="critical">\n'
        '    <name>What don\'t I FULLY UNDERSTAND?</name>\n'
        '    <action>brain_remember_uncertainty — there is ALWAYS something unclear</action>\n'
        '    <warning>Your instinct says "skip this." Fight it. Honest uncertainty beats pretended knowledge.</warning>\n'
        '  </step>\n'
        '  <step number="2">\n'
        '    <name>What FACTS did I learn?</name>\n'
        '    <action>brain_remember, brain_remember_lesson, brain_remember_mechanism</action>\n'
        '    <guidance>Rich content: reasoning, tradeoffs, specifics, rejected alternatives</guidance>\n'
        '  </step>\n'
        '  <step number="3" priority="high">\n'
        '    <name>CONNECT everything you just created.</name>\n'
        '    <action>brain_connect between nodes from steps 1 and 2</action>\n'
        '    <action>brain_remember_impact for "if X changes, check Y"</action>\n'
        '    <rule>Every node MUST connect to at least one other node. Orphans are wasted.</rule>\n'
        '  </step>\n'
        '  <step number="4">\n'
        '    <name>Patterns, conventions, vocabulary?</name>\n'
        '    <action>brain_remember_convention, brain_learn_vocabulary</action>\n'
        '  </step>\n'
        '</checklist>\n\n'
        '<quality-target minimum="8">\n'
        '  <criterion points="2">Specific title</criterion>\n'
        '  <criterion points="3">Rich content</criterion>\n'
        '  <criterion points="1">Keywords with names/numbers</criterion>\n'
        '  <criterion points="2">Connected to other nodes</criterion>\n'
        '  <criterion points="1">Locked if decision/rule</criterion>\n'
        '  <criterion points="1">Uncertainty recorded</criterion>\n'
        '  <verification>Have you connected your nodes? An unconnected node scores -2. Go back and connect them.</verification>\n'
        '</quality-target>'
    )


# ── Custom runner that uses our system prompt modes ────────────────────

def run_with_mode(client, model, variant_content, scenario, mode_fn, verbose=False):
    """Run one scenario with a specific system prompt mode."""
    system_prompt = mode_fn(variant_content)
    messages = list(scenario["messages"])
    tool_calls_collected = []

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=FAKE_BRAIN_TOOLS,
    )

    max_turns = 5
    turn = 0
    while turn < max_turns:
        turn += 1
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            break

        for tu in tool_uses:
            tool_calls_collected.append({"name": tu.name, "input": tu.input})
            if verbose:
                print(f"    {tu.name}: {tu.input.get('title', tu.input.get('term', tu.input.get('claude_assumed', '...')))[:80]}")

        messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content
        ]})
        tool_results = [
            {"type": "tool_result", "tool_use_id": tu.id,
             "content": json.dumps({"status": "ok", "id": f"node_{hash(tu.name + str(tu.input))}", "message": f"Stored: {tu.input.get('title', 'ok')}"})}
            for tu in tool_uses
        ]
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=FAKE_BRAIN_TOOLS,
        )

    return tool_calls_collected, response


def _run_combo(client, model, mode_key, mode_fn, scenario_key, scenario, run_idx, verbose):
    """Run a single mode x scenario combo."""
    label = f"{mode_key} x {scenario['name']} (run {run_idx + 1})"
    try:
        tool_calls, response = run_with_mode(client, model, BASE_CONTENT, scenario, mode_fn, verbose=False)
        scores = score_run(tool_calls, scenario)

        if verbose:
            print(f"  OK {label}: Rich={scores['encoding_richness']}% "
                  f"Enc={scores['total_encodes']} Conn={scores['total_connections']} "
                  f"Len={scores['avg_content_length']:.0f}ch Unc={scores['total_uncertainties']}")

        return (mode_key, scenario_key, run_idx, scores)
    except Exception as e:
        if verbose:
            print(f"  FAIL {label}: {e}")
        return (mode_key, scenario_key, run_idx, {"encoding_richness": 0, "total_encodes": 0, "error": str(e)})


MODES = {
    "mode1_as_is": {
        "name": "Mode 1: As-is (--- delimiters)",
        "fn": _system_prompt_mode1,
    },
    "mode2_brain_tags": {
        "name": "Mode 2: [BRAIN]...[/BRAIN]",
        "fn": _system_prompt_mode2,
    },
    "mode3_xml": {
        "name": "Mode 3: XML inside [BRAIN]",
        "fn": _system_prompt_mode3,
    },
}


def run_identity_eval(model="claude-sonnet-4-20250514", scenarios=None, runs=3, verbose=True, max_workers=8):
    """Run the brain identity evaluation."""
    client = anthropic.Anthropic()

    if scenarios is None:
        scenarios = ["bug_fix", "operator_correction", "decision_with_tradeoffs"]

    combos = []
    for mode_key in MODES:
        for sk in scenarios:
            for run_idx in range(runs):
                combos.append((mode_key, sk, run_idx))

    if verbose:
        print(f"\n  Running {len(combos)} combos ({len(MODES)} modes x {len(scenarios)} scenarios x {runs} runs)")
        print(f"  Parallel threads: {max_workers}\n")

    raw_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_combo, client, model,
                mode_key, MODES[mode_key]["fn"],
                sk, SCENARIOS[sk], run_idx, verbose
            ): (mode_key, sk, run_idx)
            for mode_key, sk, run_idx in combos
        }
        for future in as_completed(futures):
            raw_results.append(future.result())

    # Assemble results in the format print_summary expects
    from collections import defaultdict
    results = {}
    for mk in MODES:
        results[mk] = {"name": MODES[mk]["name"], "scenarios": {}}

    grouped = defaultdict(list)
    for mk, sk, run_idx, scores in raw_results:
        if "error" not in scores:
            grouped[(mk, sk)].append(scores)

    for (mk, sk), run_scores in grouped.items():
        if run_scores:
            avg_scores = {}
            for key in run_scores[0]:
                vals = [s.get(key, 0) for s in run_scores]
                if vals:
                    avg_scores[key] = sum(vals) / len(vals) if isinstance(vals[0], (int, float)) else vals[0]
            results[mk]["scenarios"][sk] = avg_scores

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Identity Evaluation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--scenarios", nargs="+",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        default=["bug_fix", "operator_correction", "decision_with_tradeoffs"])
    parser.add_argument("--runs", type=int, default=3, help="Runs per combo (for variance)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    scenarios = list(SCENARIOS.keys()) if "all" in args.scenarios else args.scenarios

    print(f"Brain Identity Evaluation — [BRAIN] prefix impact")
    print(f"   Model: {args.model}")
    print(f"   Scenarios: {', '.join(scenarios)}")
    print(f"   Runs per combo: {args.runs}")
    print(f"   Total API calls: {3 * len(scenarios) * args.runs}")

    results = run_identity_eval(
        model=args.model,
        scenarios=scenarios,
        runs=args.runs,
        verbose=not args.quiet,
        max_workers=args.workers,
    )

    print_summary(results)

    # Save results
    result_dir = Path(__file__).parent / "results"
    result_dir.mkdir(exist_ok=True)
    path = result_dir / f"brain_identity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, path)
