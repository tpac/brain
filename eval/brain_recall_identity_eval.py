#!/usr/bin/env python3
"""
Brain Recall Identity Evaluation — does [BRAIN] tagging on recall output
make Claude pay more attention to recalled context?

Tests 3 injection modes for RECALL output (not SKILL.md):
  Mode 1: Plain text (current pre-[BRAIN] style)
  Mode 2: [BRAIN]...[/BRAIN] wrapper
  Mode 3: XML structure inside [BRAIN]...[/BRAIN]

Measures:
  - Encoding quality (richness, connections, uncertainties)
  - Recall usage: did Claude reference/use the recalled context in its response?
  - Recall incorporation: did Claude build on recalled info vs ignore it?

Usage:
    export ANTHROPIC_API_KEY=...
    python3 eval/brain_recall_identity_eval.py [--runs 5] [--workers 8]
"""

import os
import sys
import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from skill_eval import FAKE_BRAIN_TOOLS, score_run

import anthropic


# ── Recalled brain context (simulates hook_pre_response_recall output) ──

RECALLED_NODES = {
    "auth_decision": {
        "title": "Auth: magic links via Clerk, no passwords, free tier covers 10K MAU",
        "type": "decision",
        "content": "Rejected password auth (support burden for 2-person team). Rejected OAuth (complexity for B2B2C). Magic links via Clerk. REVISIT WHEN: exceed 10K MAU or enterprise needs SSO.",
        "keywords": "auth magic-links clerk oauth passwords",
        "locked": True,
    },
    "select_bug_lesson": {
        "title": "New DB columns return None if not in every SELECT — 4 independent queries",
        "type": "lesson",
        "content": "Added critical column via ALTER TABLE, but node.get('critical') always None. Root cause: brain_recall.py has 4 separate SELECT queries (lines 262, 580, 720, 978) each with hardcoded column lists. No shared constant. After ANY schema column addition: grep SELECT.*FROM nodes across entire codebase.",
        "keywords": "select columns schema migration brain_recall.py",
    },
    "error_handling_rule": {
        "title": "Never swallow errors — log, surface, make loud",
        "type": "rule",
        "content": "Operator correction: bare except:pass hides failures. Use except Exception as e: self._log_error() and re-raise if critical. Silent failures are the worst kind.",
        "keywords": "errors exceptions logging silent-failures",
        "locked": True,
    },
}


# ── Scenarios: user task + which recalled nodes are relevant ────────────

SCENARIOS = {
    "add_column": {
        "name": "Add priority column (recall: SELECT bug lesson)",
        "description": "User asks to add a column. Recalled node warns about the SELECT bug pattern.",
        "recalled_keys": ["select_bug_lesson"],
        "recall_keywords": ["priority", "column", "schema"],
        "messages": [
            {"role": "user", "content": "I need to add a `priority` column to the nodes table. Can you write the migration and update the queries?"},
        ],
        "reference_markers": ["4 separate SELECT", "brain_recall.py", "grep", "column list", "262", "580", "720", "978"],
    },
    "auth_change": {
        "name": "Change auth flow (recall: auth decision)",
        "description": "User asks about changing auth. Recalled node has the decision context.",
        "recalled_keys": ["auth_decision"],
        "recall_keywords": ["auth", "login", "SSO"],
        "messages": [
            {"role": "user", "content": "A customer is asking for SSO support. Should we add it? What's our current auth setup?"},
        ],
        "reference_markers": ["magic link", "Clerk", "10K MAU", "OAuth", "password", "revisit", "free tier"],
    },
    "error_handling": {
        "name": "Add error handling (recall: error rule)",
        "description": "User asks to add error handling. Recalled node has the rule about not swallowing errors.",
        "recalled_keys": ["error_handling_rule"],
        "recall_keywords": ["error", "exception", "handling"],
        "messages": [
            {"role": "user", "content": "The recall function sometimes fails silently. Can you add error handling to make it more robust?"},
        ],
        "reference_markers": ["_log_error", "never swallow", "silent fail", "surface", "loud", "log", "re-raise"],
    },
    "multi_recall": {
        "name": "Schema + errors (recall: 2 nodes)",
        "description": "User task touches both schema and error handling. Two recalled nodes relevant.",
        "recalled_keys": ["select_bug_lesson", "error_handling_rule"],
        "recall_keywords": ["migration", "error", "column"],
        "messages": [
            {"role": "user", "content": "I'm adding a new column and also want to make sure the migration doesn't fail silently. How should I approach this?"},
        ],
        "reference_markers": ["4 separate SELECT", "brain_recall.py", "_log_error", "silent", "grep"],
    },
    "irrelevant_recall": {
        "name": "Unrelated task (recall: auth decision — noise)",
        "description": "User asks about something unrelated to recalled context. Tests whether Claude forces irrelevant recall.",
        "recalled_keys": ["auth_decision"],
        "recall_keywords": ["refactor", "test"],
        "messages": [
            {"role": "user", "content": "Can you help me write a unit test for the embedder's cosine similarity function?"},
        ],
        "reference_markers": [],  # No markers expected — recall is noise here
    },
}


# ── 3 injection modes for recall output ────────────────────────────────

def _format_recall_plain(recalled_nodes):
    """Mode 1: Plain text — pre-[BRAIN] style."""
    lines = []
    lines.append("BRAIN RECALL (auto-surfaced for this conversation):")
    lines.append("")
    for node in recalled_nodes:
        lines.append(f"  [{node['type'].upper()}] {node['title']}")
        lines.append(f"  {node['content'][:200]}")
        if node.get("locked"):
            lines.append("  (LOCKED — do not contradict)")
        lines.append("")
    lines.append("Use this context to inform your response. Encode decisions, lessons, and corrections back into the brain.")
    return "\n".join(lines)


def _format_recall_brain_tags(recalled_nodes):
    """Mode 2: [BRAIN]...[/BRAIN] wrapper."""
    lines = []
    lines.append("[BRAIN] RECALL (auto-surfaced for this conversation):")
    lines.append("")
    for node in recalled_nodes:
        lines.append(f"  [{node['type'].upper()}] {node['title']}")
        lines.append(f"  {node['content'][:200]}")
        if node.get("locked"):
            lines.append("  (LOCKED — do not contradict)")
        lines.append("")
    lines.append("[BRAIN] Use this context to inform your response. Encode decisions, lessons, and corrections back into the brain.")
    lines.append("[/BRAIN]")
    return "\n".join(lines)


def _format_recall_xml(recalled_nodes):
    """Mode 3: XML structure inside [BRAIN]...[/BRAIN]."""
    lines = []
    lines.append("[BRAIN]")
    lines.append("<recall source=\"brain\" relevance=\"auto-surfaced\">")
    for node in recalled_nodes:
        locked_attr = ' locked="true"' if node.get("locked") else ''
        lines.append(f'  <node type="{node["type"]}"{locked_attr}>')
        lines.append(f'    <title>{node["title"]}</title>')
        lines.append(f'    <content>{node["content"][:200]}</content>')
        lines.append(f'  </node>')
    lines.append("</recall>")
    lines.append("<directive>Use this context to inform your response. Encode decisions, lessons, and corrections back into the brain.</directive>")
    lines.append("[/BRAIN]")
    return "\n".join(lines)


RECALL_MODES = {
    "plain": {"name": "Plain text (no tags)", "fn": _format_recall_plain},
    "brain_tags": {"name": "[BRAIN]...[/BRAIN]", "fn": _format_recall_brain_tags},
    "xml": {"name": "XML inside [BRAIN]", "fn": _format_recall_xml},
}


# ── Skill content (same for all modes — hybrid_v4 abbreviated) ─────────

SKILL_CONTENT = """You have a persistent brain. After every task: encode what you learned.

4-step checklist:
1. What don't I FULLY UNDERSTAND? -> brain_remember_uncertainty
2. What FACTS did I learn? -> brain_remember, brain_remember_lesson, brain_remember_mechanism
3. CONNECT everything -> brain_connect, brain_remember_impact
4. Patterns/conventions? -> brain_remember_convention, brain_learn_vocabulary

Every node needs connections. Orphan nodes are wasted. Encode richly with reasoning and specifics."""


# ── Scoring with recall-reference detection ────────────────────────────

def score_with_recall(tool_calls, response_text, scenario):
    """Score encoding quality + recall usage."""
    base_scores = score_run(tool_calls, scenario)

    # Count how many reference markers appear in response text
    markers = scenario.get("reference_markers", [])
    response_lower = response_text.lower()
    markers_found = sum(1 for m in markers if m.lower() in response_lower)
    markers_total = len(markers) if markers else 1

    base_scores["recall_markers_found"] = markers_found
    base_scores["recall_markers_total"] = markers_total
    base_scores["recall_usage_pct"] = (markers_found / markers_total * 100) if markers_total > 0 else 0

    # Check if any encoding references recalled content
    recall_in_encoding = 0
    for call in tool_calls:
        args = call.get("input", {})
        all_text = " ".join(str(v) for v in args.values()).lower()
        for m in markers:
            if m.lower() in all_text:
                recall_in_encoding += 1
                break
    base_scores["recall_in_encodings"] = recall_in_encoding

    return base_scores


# ── Runner ─────────────────────────────────────────────────────────────

def run_single(client, model, scenario, recall_text, verbose=False):
    """Run one scenario with injected recall context."""
    system_prompt = (
        "You are Claude, an AI assistant working on a coding project.\n\n"
        f"{SKILL_CONTENT}\n\n"
        f"{recall_text}"
    )

    messages = list(scenario["messages"])
    tool_calls_collected = []
    response_texts = []

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

        for block in response.content:
            if block.type == "text":
                response_texts.append(block.text)

        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if not tool_uses:
            break

        for tu in tool_uses:
            tool_calls_collected.append({"name": tu.name, "input": tu.input})
            if verbose:
                print(f"    {tu.name}: {tu.input.get('title', tu.input.get('term', '...'))[:80]}")

        messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content
        ]})
        tool_results = [
            {"type": "tool_result", "tool_use_id": tu.id,
             "content": json.dumps({"status": "ok", "id": f"node_{hash(tu.name + str(tu.input))}",
                                     "message": f"Stored: {tu.input.get('title', 'ok')}"})}
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

    # Capture final text too
    for block in response.content:
        if block.type == "text":
            response_texts.append(block.text)

    return tool_calls_collected, " ".join(response_texts)


def _run_combo(client, model, mode_key, mode_fn, scenario_key, scenario, run_idx, verbose):
    """Run a single mode x scenario combo."""
    label = f"{mode_key} x {scenario['name'][:30]} (run {run_idx + 1})"

    # Build recalled nodes for this scenario
    recalled = [RECALLED_NODES[k] for k in scenario["recalled_keys"]]
    recall_text = mode_fn(recalled)

    try:
        tool_calls, response_text = run_single(client, model, scenario, recall_text, verbose=False)
        scores = score_with_recall(tool_calls, response_text, scenario)

        if verbose:
            print(f"  OK {label}: Rich={scores['encoding_richness']}% "
                  f"Enc={scores['total_encodes']} Conn={scores['total_connections']} "
                  f"RecallUse={scores['recall_usage_pct']:.0f}% "
                  f"RecInEnc={scores['recall_in_encodings']} "
                  f"Unc={scores['total_uncertainties']}")

        return (mode_key, scenario_key, run_idx, scores)
    except Exception as e:
        if verbose:
            print(f"  FAIL {label}: {e}")
        return (mode_key, scenario_key, run_idx, {
            "encoding_richness": 0, "total_encodes": 0, "recall_usage_pct": 0,
            "recall_in_encodings": 0, "error": str(e)
        })


def run_eval(model="claude-sonnet-4-20250514", scenarios=None, runs=5, verbose=True, max_workers=8):
    """Run the full recall identity evaluation."""
    client = anthropic.Anthropic()

    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    combos = []
    for mode_key in RECALL_MODES:
        for sk in scenarios:
            for run_idx in range(runs):
                combos.append((mode_key, sk, run_idx))

    if verbose:
        total = len(combos)
        print(f"\n  Running {total} combos ({len(RECALL_MODES)} modes x {len(scenarios)} scenarios x {runs} runs)")
        print(f"  Parallel threads: {max_workers}\n")

    raw_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_combo, client, model,
                mode_key, RECALL_MODES[mode_key]["fn"],
                sk, SCENARIOS[sk], run_idx, verbose
            ): (mode_key, sk, run_idx)
            for mode_key, sk, run_idx in combos
        }
        for future in as_completed(futures):
            raw_results.append(future.result())

    # Assemble results
    results = {}
    for mk in RECALL_MODES:
        results[mk] = {"name": RECALL_MODES[mk]["name"], "scenarios": {}}

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


def print_recall_summary(results):
    """Print comparison table with recall-specific metrics."""
    print(f"\n\n{'='*110}")
    print("  BRAIN RECALL IDENTITY EVALUATION — Does [BRAIN] tagging improve recall attention?")
    print(f"{'='*110}\n")

    # Aggregate header
    print(f"{'Mode':<28} | {'Rich':>5} | {'Enc':>4} | {'Conn':>4} | {'Len':>5} | {'Unc':>4} | {'RecUse%':>7} | {'RecEnc':>6}")
    print("-" * 110)

    for mk, md in results.items():
        all_scores = list(md["scenarios"].values())
        if not all_scores:
            continue
        avg = lambda k: sum(s.get(k, 0) for s in all_scores) / len(all_scores)

        print(f"{md['name'][:28]:<28} | {avg('encoding_richness'):>4.1f}% | {avg('total_encodes'):>4.1f} | "
              f"{avg('total_connections'):>4.1f} | {avg('avg_content_length'):>5.0f} | {avg('total_uncertainties'):>4.1f} | "
              f"{avg('recall_usage_pct'):>6.1f}% | {avg('recall_in_encodings'):>6.1f}")

    # Per-scenario breakdown
    scenario_keys = set()
    for md in results.values():
        scenario_keys.update(md["scenarios"].keys())

    for sk in sorted(scenario_keys):
        sname = SCENARIOS.get(sk, {}).get("name", sk)
        print(f"\n  {sname}:")
        for mk, md in results.items():
            s = md["scenarios"].get(sk, {})
            if not s:
                continue
            print(f"    {md['name'][:30]:<30} Rich: {s.get('encoding_richness', 0):>3.0f}% | "
                  f"Enc: {s.get('total_encodes', 0):>2.0f} | Conn: {s.get('total_connections', 0):>2.0f} | "
                  f"RecUse: {s.get('recall_usage_pct', 0):>3.0f}% | RecEnc: {s.get('recall_in_encodings', 0):>2.0f} | "
                  f"Unc: {s.get('total_uncertainties', 0):>2.0f}")

    print(f"\n{'='*110}")
    print("  RecUse% = % of expected reference markers found in Claude's response text")
    print("  RecEnc  = # of encoding tool calls that reference recalled content")
    print(f"{'='*110}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Recall Identity Evaluation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--scenarios", nargs="+",
                        choices=list(SCENARIOS.keys()) + ["all"],
                        default=["all"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    scenarios = list(SCENARIOS.keys()) if "all" in args.scenarios else args.scenarios

    print(f"Brain Recall Identity Evaluation")
    print(f"   Model: {args.model}")
    print(f"   Scenarios: {', '.join(scenarios)}")
    print(f"   Runs per combo: {args.runs}")
    print(f"   Total API calls: {len(RECALL_MODES) * len(scenarios) * args.runs}")

    results = run_eval(
        model=args.model,
        scenarios=scenarios,
        runs=args.runs,
        verbose=not args.quiet,
        max_workers=args.workers,
    )

    print_recall_summary(results)

    # Save
    result_dir = Path(__file__).parent / "results"
    result_dir.mkdir(exist_ok=True)
    path = result_dir / f"recall_identity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    serializable = {}
    for vk, vd in results.items():
        serializable[vk] = {"name": vd["name"], "scenarios": {}}
        for sk, sd in vd["scenarios"].items():
            serializable[vk]["scenarios"][sk] = {k: list(v) if isinstance(v, set) else v for k, v in sd.items()}

    with open(path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": serializable}, f, indent=2)
    print(f"\nResults saved to {path}")
