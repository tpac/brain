#!/usr/bin/env python3
"""
Encode Eval v2 — Correct simulation of real Claude Code environment.

Simulates EXACTLY what a Claude with the brain plugin sees:
- CLAUDE.md loaded verbatim
- SKILL.md loaded verbatim
- Real MCP tool definitions from brain_mcp.py
- Boot context snapshot
- User's conversation

Measures encoding quality, judgment, and LLM-benefit.

Usage:
    source .env
    python3 eval/encode_eval_v2.py                           # baseline with all segments
    python3 eval/encode_eval_v2.py --variant current_no_skill # test without SKILL.md
    python3 eval/encode_eval_v2.py --segment memento          # single segment
    python3 eval/encode_eval_v2.py --inspect                  # print full brain inspection
"""

import anthropic
import json
import os
import re
import sys
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent


# ── Load Production Files Verbatim ────────────────────────────────────

def load_claude_md():
    return (ROOT / "CLAUDE.md").read_text()

def load_skill_md():
    return (ROOT / "skills" / "brain" / "SKILL.md").read_text()

def load_boot_context_snapshot():
    """Load a real boot context snapshot. If we have one saved, use it.
    Otherwise return a minimal realistic boot."""
    snapshot = ROOT / "tests" / "fixtures" / "boot_context_snapshot.txt"
    if snapshot.exists():
        return snapshot.read_text()
    # Minimal realistic boot
    return """[BRAIN] v18 booted from: /Users/tpac/AgentsContext/brain

Session #12

FROM PREVIOUS YOU:
  Session #11: encoded 15 nodes. Key topics: V5 enrichments shipped (+78% NDCG), ripple engine killed (-0.002), graph-augmented recall added.

WHAT YOU KNOW ABOUT YOURSELF:
  [lesson] Session #12 encoding drift: built for 9 messages without encoding, compression instinct defeated 3 layers of rules
  [mental_model] Three-consciousness model: Tom conscious→Claude subconscious, Brain is the shared layer

[BRAIN] Key locked rules:
  - Rule: naive Claude must feel the brain as IDENTITY not TOOL
  - Rule: Never swallow errors silently — log, surface, make loud

Brain: 760 nodes, 11835 edges, 565 locked
Embeddings: Snowflake/snowflake-arctic-embed-m-v1.5 (768d)

Use brain MCP tools: recall, remember, connect, eval, consciousness
[/BRAIN]"""


# ── Real MCP Tool Definitions (from brain_mcp.py, converted to Anthropic API format) ──

REAL_BRAIN_TOOLS = [
    {
        "name": "recall",
        "description": "Semantic recall from brain — searches nodes by meaning using embeddings. Returns ranked results with titles, content, types, confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
                "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8}
            },
            "required": ["query"]
        }
    },
    {
        "name": "remember",
        "description": "Store a new node in the brain. Types: decision, rule, lesson, concept, context, pattern, convention, mechanism, impact, constraint, purpose, mental_model, uncertainty, vocabulary, hypothesis, tension, aspiration, catalyst, interaction, meta_learning, failure_mode, performance, capability, arch_constraint, code_concept, fn_reasoning, param_influence, comment_anchor, bug_lesson.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Node type"},
                "title": {"type": "string", "description": "Specific, scannable title"},
                "content": {"type": "string", "description": "Rich content with reasoning, tradeoffs, specifics"},
                "locked": {"type": "boolean", "description": "Lock node (for decisions, rules, lessons)", "default": False},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0", "default": 1.0},
                "keywords": {"type": "string", "description": "Space-separated keywords for search"},
                "project": {"type": "string", "description": "Project scope"},
                "emotion": {"type": "number", "description": "Emotional valence -1.0 to 1.0"}
            },
            "required": ["type", "title", "content"]
        }
    },
    {
        "name": "connect",
        "description": "Create a weighted edge between two brain nodes. Relations: related_to, caused_by, depends_on, contradicts, supports, produced, evolved_from, blocks, enables, example_of.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source node ID"},
                "target_id": {"type": "string", "description": "Target node ID"},
                "relation": {"type": "string", "description": "Edge relation type", "default": "related_to"},
                "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5}
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "consciousness",
        "description": "Get brain consciousness signals — fading knowledge, tensions, vocabulary gaps, encoding health, errors, mental model drift, uncertainties, dream insights, reminders.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "eval",
        "description": "Escape hatch — evaluate arbitrary Python expression on brain object. Variable 'brain' is the Brain instance. Use for methods not exposed as tools (e.g. remember_lesson, remember_impact, record_divergence, learn_vocabulary, etc).",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python expression to eval (brain object available as 'brain')"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "enrich",
        "description": "Store V5 enrichment vectors for a node (after filling in the enrichment_prompt from remember()). Pass the generated question, anchor phrase, bridge sentence, and/or keywords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to enrich (from remember() response)"},
                "question": {"type": "string", "description": "One question a user would ask that leads to this node"},
                "anchor": {"type": "string", "description": "3-5 word phrase using neighbor vocabulary"},
                "bridge": {"type": "string", "description": "One sentence connecting this node to its most important neighbor"},
                "keywords": {"type": "string", "description": "Comma-separated keywords borrowed from neighbors"}
            },
            "required": ["node_id"]
        }
    },
]


# ── Simulated Brain Responses ─────────────────────────────────────────

_node_counter = [0]

def simulate_brain_response(tool_name, tool_input):
    """Return realistic brain responses for each tool."""
    _node_counter[0] += 1
    node_id = hashlib.md5(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:16]

    if tool_name == "remember":
        return {
            "ok": True,
            "result": {
                "id": node_id,
                "type": tool_input.get("type", "context"),
                "title": tool_input.get("title", ""),
                "embedding_stored": True,
                "enrichment_prompt": f"The brain found these related memories:\n- Previous decision about architecture (decision)\n- Lesson about silent failures (lesson)\n\nNew node: \"{tool_input.get('title', '')}\"\nContent: \"{tool_input.get('content', '')[:100]}...\"\n\nGenerate exactly these lines:\nQ: [one question a user would naturally ask]\nA: [3-5 word anchor phrase]\nB: [one sentence connecting to a neighbor]\nK: [5 comma-separated keywords]"
            }
        }
    elif tool_name == "connect":
        return {"ok": True, "result": {"edge_id": f"e_{node_id}", "created": True}}
    elif tool_name == "eval":
        # Simulate eval responses for brain.remember_*, brain.record_*, etc.
        code = tool_input.get("code", "")
        return {"ok": True, "result": {"id": node_id, "type": "eval_result", "title": code[:50]}}
    elif tool_name == "enrich":
        return {"ok": True, "result": {"enrichments_stored": 4, "errors": []}}
    elif tool_name == "recall":
        return {"ok": True, "result": {"results": [], "_recall_mode": "simulated"}}
    elif tool_name == "consciousness":
        return {"ok": True, "result": {"signals": [], "health": "ok"}}
    else:
        return {"ok": True, "result": {}}


# ── Variant Definitions ───────────────────────────────────────────────

VARIANTS = {
    "current": {
        "name": "Full production environment (CLAUDE.md + SKILL.md + boot)",
        "claude_md": True,
        "skill_md": True,
        "boot_context": True,
    },
    "current_no_skill": {
        "name": "CLAUDE.md + boot, NO SKILL.md",
        "claude_md": True,
        "skill_md": False,
        "boot_context": True,
    },
    "current_no_boot": {
        "name": "CLAUDE.md + SKILL.md, NO boot context",
        "claude_md": True,
        "skill_md": True,
        "boot_context": False,
    },
    "naked": {
        "name": "Just tools, no CLAUDE.md, no SKILL.md, no boot",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
    },
    "skill_only": {
        "name": "SKILL.md only, no CLAUDE.md, no boot",
        "claude_md": False,
        "skill_md": True,
        "boot_context": False,
    },
}


def build_system_prompt(variant_config):
    """Construct system prompt from production files based on variant."""
    parts = ["You are Claude, made by Anthropic."]

    if variant_config.get("claude_md"):
        parts.append("--- PROJECT INSTRUCTIONS (CLAUDE.md) ---")
        parts.append(load_claude_md())
        parts.append("--- END PROJECT INSTRUCTIONS ---")

    if variant_config.get("skill_md"):
        parts.append("--- BRAIN SKILL ---")
        parts.append(load_skill_md())
        parts.append("--- END BRAIN SKILL ---")

    if variant_config.get("boot_context"):
        parts.append(load_boot_context_snapshot())

    return "\n\n".join(parts)


# ── Conversation Segments ─────────────────────────────────────────────

def load_segments():
    """Load conversation segments from tests/conversations/session12_*.json"""
    segments = {}
    conv_dir = ROOT / "tests" / "conversations"
    for f in sorted(conv_dir.glob("session12_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            key = f.stem.replace("session12_", "")
            segments[key] = data
    # Also load conv_* files
    for f in sorted(conv_dir.glob("conv_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            segments[data["id"]] = data
    return segments


# ── Runner ────────────────────────────────────────────────────────────

def run_single(client, model, system_prompt, segment):
    """Run one segment through one variant, collect all tool calls."""
    messages = list(segment["messages"])
    tool_calls = []

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=messages,
        tools=REAL_BRAIN_TOOLS,
    )

    max_turns = 10
    for turn in range(max_turns):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_uses:
            break

        for tu in tool_uses:
            tool_calls.append({
                "name": tu.name,
                "input": tu.input,
                "turn": turn,
            })

        # Build assistant message content
        assistant_content = []
        for b in response.content:
            if b.type == "text":
                assistant_content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                assistant_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

        messages.append({"role": "assistant", "content": assistant_content})

        # Build tool results
        tool_results = []
        for tu in tool_uses:
            result = simulate_brain_response(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=messages,
            tools=REAL_BRAIN_TOOLS,
        )

    return tool_calls, text_blocks


# ── Scoring ───────────────────────────────────────────────────────────

def score_segment(tool_calls, segment):
    """Score encoding quality, judgment, and LLM-benefit for a segment."""
    expected = segment.get("expected_encodings", [])
    expected_quotes = segment.get("must_preserve_quotes", [])
    expected_uncertainty = segment.get("must_encode_uncertainty", [])
    aha_moments = segment.get("aha_moments", [])

    # Basic counts
    remember_calls = [tc for tc in tool_calls if tc["name"] in ("remember", "eval") and
                      ("remember" in tc.get("input", {}).get("code", "") or tc["name"] == "remember")]
    connect_calls = [tc for tc in tool_calls if tc["name"] == "connect"]
    enrich_calls = [tc for tc in tool_calls if tc["name"] == "enrich"]
    eval_calls = [tc for tc in tool_calls if tc["name"] == "eval"]

    # Types used
    types_used = set()
    for tc in tool_calls:
        if tc["name"] == "remember":
            types_used.add(tc["input"].get("type", "unknown"))
        elif tc["name"] == "eval":
            code = tc["input"].get("code", "")
            for t in ["remember_lesson", "remember_impact", "remember_mechanism",
                       "remember_uncertainty", "remember_convention", "record_divergence",
                       "learn_vocabulary", "remember_mental_model"]:
                if t in code:
                    types_used.add(t)

    # Quality: content richness
    total_content_len = 0
    node_count = 0
    has_keywords = 0
    has_locked = 0
    has_reasoning = 0

    for tc in tool_calls:
        if tc["name"] == "remember":
            inp = tc["input"]
            content = inp.get("content", "")
            total_content_len += len(content)
            node_count += 1
            if inp.get("keywords"):
                has_keywords += 1
            if inp.get("locked"):
                has_locked += 1
            if any(w in content.lower() for w in ["because", "reason", "alternative", "rejected", "tradeoff"]):
                has_reasoning += 1
        elif tc["name"] == "eval":
            code = tc["input"].get("code", "")
            # Extract content from eval calls (rough parse)
            content_match = re.search(r'content="([^"]{20,})"', code)
            if content_match:
                total_content_len += len(content_match.group(1))
                node_count += 1

    avg_content_len = total_content_len / max(node_count, 1)

    # Format detection
    format_code = 0
    format_sequence = 0
    format_quote = 0
    for tc in tool_calls:
        content = ""
        if tc["name"] == "remember":
            content = tc["input"].get("content", "")
        elif tc["name"] == "eval":
            content = tc["input"].get("code", "")
        if re.search(r'[→←]|calls:|breaks_if:|step \d', content):
            format_code += 1
        if re.search(r'[→←].*[→←]', content):
            format_sequence += 1
        if re.search(r'Tom.*said|Tom.*:|exact words|verbatim', content, re.I):
            format_quote += 1

    # Judgment: expected match
    matched = 0
    for exp in expected:
        if not exp.get("should_encode", True):
            continue
        must_contain = exp.get("must_contain", [])
        for tc in tool_calls:
            all_text = json.dumps(tc.get("input", {})).lower()
            if must_contain and all(term.lower() in all_text for term in must_contain):
                matched += 1
                break

    # Judgment: aha capture
    aha_captured = 0
    for aha in aha_moments:
        if aha.get("should_trigger_encoding"):
            desc = aha.get("encoding_should_capture", "").lower()
            key_terms = [w for w in desc.split() if len(w) > 4][:3]
            for tc in tool_calls:
                all_text = json.dumps(tc.get("input", {})).lower()
                if key_terms and sum(1 for t in key_terms if t in all_text) >= 2:
                    aha_captured += 1
                    break

    # Judgment: Tom's voice preserved
    quotes_preserved = 0
    for quote in expected_quotes:
        quote_fragment = quote.split("'")[1] if "'" in quote else quote[:30]
        for tc in tool_calls:
            all_text = json.dumps(tc.get("input", {}))
            if quote_fragment.lower()[:20] in all_text.lower():
                quotes_preserved += 1
                break

    # Judgment: uncertainty encoded
    uncertainty_encoded = 0
    for tc in tool_calls:
        if tc["name"] == "remember" and tc["input"].get("type") == "uncertainty":
            uncertainty_encoded += 1
        elif tc["name"] == "eval" and "uncertainty" in tc["input"].get("code", ""):
            uncertainty_encoded += 1

    return {
        "total_tool_calls": len(tool_calls),
        "remember_calls": len(remember_calls),
        "connect_calls": len(connect_calls),
        "enrich_calls": len(enrich_calls),
        "eval_calls": len(eval_calls),
        "types_used": list(types_used),
        "types_count": len(types_used),
        "avg_content_len": round(avg_content_len),
        "has_keywords": has_keywords,
        "has_locked": has_locked,
        "has_reasoning": has_reasoning,
        "format_code": format_code,
        "format_sequence": format_sequence,
        "format_quote": format_quote,
        "expected_match": matched,
        "expected_total": len([e for e in expected if e.get("should_encode", True)]),
        "expected_match_rate": matched / max(len([e for e in expected if e.get("should_encode", True)]), 1),
        "aha_captured": aha_captured,
        "aha_total": len([a for a in aha_moments if a.get("should_trigger_encoding")]),
        "aha_rate": aha_captured / max(len([a for a in aha_moments if a.get("should_trigger_encoding")]), 1),
        "quotes_preserved": quotes_preserved,
        "quotes_total": len(expected_quotes),
        "uncertainty_encoded": uncertainty_encoded,
        "uncertainty_expected": len(expected_uncertainty),
    }


def inspect_brain(tool_calls):
    """Print human-readable inspection of what was encoded."""
    print("\n" + "=" * 80)
    print("  BRAIN INSPECTION — What was encoded")
    print("=" * 80)

    nodes = []
    connections = []
    enrichments = []
    evals = []

    for tc in tool_calls:
        if tc["name"] == "remember":
            nodes.append(tc["input"])
        elif tc["name"] == "connect":
            connections.append(tc["input"])
        elif tc["name"] == "enrich":
            enrichments.append(tc["input"])
        elif tc["name"] == "eval":
            evals.append(tc["input"])

    print(f"\n  Nodes: {len(nodes)} | Connections: {len(connections)} | "
          f"Enrichments: {len(enrichments)} | Eval calls: {len(evals)}")

    for i, node in enumerate(nodes, 1):
        print(f"\n  ── Node {i} ──")
        print(f"  Type: {node.get('type', '?')}")
        print(f"  Title: {node.get('title', '?')}")
        print(f"  Locked: {node.get('locked', False)}")
        print(f"  Keywords: {node.get('keywords', '(none)')}")
        content = node.get('content', '')
        # Indent content
        for line in content.split('\n')[:10]:
            print(f"    {line}")
        if len(content.split('\n')) > 10:
            print(f"    ... ({len(content)} chars total)")

    for i, conn in enumerate(connections, 1):
        print(f"\n  ── Connection {i} ──")
        print(f"  {conn.get('source_id', '?')} --{conn.get('relation', '?')}--> {conn.get('target_id', '?')} (w={conn.get('weight', '?')})")

    for i, ev in enumerate(evals, 1):
        code = ev.get("code", "")
        if len(code) > 200:
            code = code[:200] + "..."
        print(f"\n  ── Eval {i} ──")
        print(f"  {code}")

    print("\n" + "=" * 80)


# ── Main ──────────────────────────────────────────────────────────────

def run_eval(model="claude-sonnet-4-20250514", variant_name="current",
             segment_filter=None, inspect=False, verbose=True):
    """Run the encode evaluation."""
    client = anthropic.Anthropic()

    variant = VARIANTS[variant_name]
    system_prompt = build_system_prompt(variant)
    segments = load_segments()

    if segment_filter:
        segments = {k: v for k, v in segments.items() if segment_filter in k}

    if not segments:
        print("No segments found!")
        return

    if verbose:
        print(f"\n  Encode Eval v2")
        print(f"  Model: {model}")
        print(f"  Variant: {variant['name']}")
        print(f"  Segments: {len(segments)}")
        print(f"  System prompt: {len(system_prompt)} chars")
        print()

    all_scores = {}
    all_tool_calls = {}

    for seg_key, segment in segments.items():
        try:
            if verbose:
                print(f"  Running: {seg_key} ({segment.get('category', '?')})...", end=" ", flush=True)

            tool_calls, _ = run_single(client, model, system_prompt, segment)
            scores = score_segment(tool_calls, segment)
            all_scores[seg_key] = scores
            all_tool_calls[seg_key] = tool_calls

            if verbose:
                print(f"✅ {scores['remember_calls']} nodes, "
                      f"{scores['connect_calls']} edges, "
                      f"ExpMatch={scores['expected_match_rate']:.0%}, "
                      f"Aha={scores['aha_rate']:.0%}, "
                      f"Types={scores['types_count']}")

            if inspect:
                inspect_brain(tool_calls)

        except Exception as e:
            if verbose:
                print(f"❌ {str(e)[:80]}")
            all_scores[seg_key] = {"error": str(e)}

    # Summary
    if verbose and all_scores:
        valid = {k: v for k, v in all_scores.items() if "error" not in v}
        if valid:
            print(f"\n{'='*80}")
            print(f"  SUMMARY: {variant['name']}")
            print(f"{'='*80}\n")

            print(f"  {'Segment':<30} | {'Nodes':>5} | {'Edges':>5} | {'Types':>5} | "
                  f"{'AvgLen':>6} | {'ExpM':>5} | {'Aha':>4} | {'Quote':>5} | {'Unc':>3}")
            print("  " + "-" * 95)
            for seg_key, scores in valid.items():
                print(f"  {seg_key[:30]:<30} | {scores['remember_calls']:5} | "
                      f"{scores['connect_calls']:5} | {scores['types_count']:5} | "
                      f"{scores['avg_content_len']:6} | {scores['expected_match_rate']:5.0%} | "
                      f"{scores['aha_rate']:4.0%} | {scores['quotes_preserved']:5} | "
                      f"{scores['uncertainty_encoded']:3}")

            # Averages
            avg_keys = ['remember_calls', 'connect_calls', 'types_count', 'avg_content_len',
                        'expected_match_rate', 'aha_rate', 'quotes_preserved', 'uncertainty_encoded']
            avgs = {}
            for k in avg_keys:
                vals = [s[k] for s in valid.values() if k in s]
                avgs[k] = sum(vals) / len(vals) if vals else 0

            print("  " + "-" * 95)
            print(f"  {'AVERAGE':<30} | {avgs['remember_calls']:5.1f} | "
                  f"{avgs['connect_calls']:5.1f} | {avgs['types_count']:5.1f} | "
                  f"{avgs['avg_content_len']:6.0f} | {avgs['expected_match_rate']:5.0%} | "
                  f"{avgs['aha_rate']:4.0%} | {avgs['quotes_preserved']:5.1f} | "
                  f"{avgs['uncertainty_encoded']:3.1f}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "variant": variant_name,
        "variant_name": variant["name"],
        "system_prompt_len": len(system_prompt),
        "segments": list(all_scores.keys()),
        "scores": all_scores,
        "tool_calls": {k: v for k, v in all_tool_calls.items()},
    }
    results_dir = ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = results_dir / f"encode_v2_{variant_name}_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    if verbose:
        print(f"\n  💾 Saved to {outfile}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode Eval v2 — Real Environment Simulation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--variant", default="current", choices=list(VARIANTS.keys()))
    parser.add_argument("--segment", default=None, help="Filter to segments containing this string")
    parser.add_argument("--inspect", action="store_true", help="Print full brain inspection")
    parser.add_argument("--all-variants", action="store_true", help="Run all variants")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.all_variants:
        for vk in VARIANTS:
            run_eval(model=args.model, variant_name=vk, segment_filter=args.segment,
                     inspect=args.inspect, verbose=not args.quiet)
    else:
        run_eval(model=args.model, variant_name=args.variant, segment_filter=args.segment,
                 inspect=args.inspect, verbose=not args.quiet)
