#!/usr/bin/env python3
"""Encoding Agent v2 vs v3 — Side-by-side comparison.

Runs the same conversation through both v2 and v3 encoding prompts,
using the real brain for recall context. Compares:
- What was encoded (actions, node types, richness)
- Tool usage patterns (specialized tools vs generic remember)
- Whether v3 detects concept gaps
- Whether v3 produces a usable encoding journal
- Whether pre-attached recall changes encoding decisions

Uses InstrumentedBrain from capabilities framework for action capture.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 eval/encoding_v3_compare.py

    # Specific conversation corpus
    python3 eval/encoding_v3_compare.py --corpus eval/corpus/conv_001_architecture.json

    # Use fixture brain (default) or live brain
    python3 eval/encoding_v3_compare.py --live-brain
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

# Load .env for API key
_env_path = ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ[_k.strip()] = _v.strip()

import anthropic
from eval.capabilities.base import (
    InstrumentedBrain, dispatch_tool, CapturedAction,
    CAPABILITY_TOOLS, _build_capability_tools,
)
from eval.corpus.loader import load_corpus


# ── Prompt loaders ──

def _load_v2_prompt():
    path = ROOT / "hooks" / "prompts" / "encoding-agent.md"
    with open(path) as f:
        prompt = f.read()
    try:
        from servers.contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception:
        pass
    return prompt


def _load_v3_prompt():
    path = ROOT / "hooks" / "prompts" / "encoding-agent-v3.md"
    with open(path) as f:
        prompt = f.read()
    try:
        from servers.contract import generate_field_summary
        prompt += "\n\n## Available Fields (from contract)\n\n" + generate_field_summary()
    except Exception:
        pass
    return prompt


# ── Tool sets ──

V2_TOOLS = CAPABILITY_TOOLS  # 13 tools (includes specialized remember_*)

V3_TOOL_NAMES = {
    'recall', 'find_node_by_title', 'get_node',
    'remember', 'revise', 'connect',
    'record_divergence', 'learn_vocabulary',
}
V3_TOOLS = [t for t in CAPABILITY_TOOLS if t["name"] in V3_TOOL_NAMES]


# ── User content builders ──

def _build_v2_content(conversation: dict, brain, counter: int = 5) -> str:
    """Build v2-style flat user content."""
    exchanges = conversation["exchanges"]

    msg_text = ""
    for ex in exchanges:
        role = ex["role"].upper()
        content = ex["content"][:600]
        msg_text += "[%s]: %s\n\n" % (role, content)

    # Independent recall (mimics v2 _gather_recall_context)
    user_msgs = [ex["content"] for ex in exchanges if ex["role"] == "user"]
    recall_query = " ".join(msg[:200] for msg in user_msgs[-3:])
    recall_context = ""
    try:
        from servers.brain_voice import BrainVoice
        result = brain._brain.recall(query=recall_query, limit=5)
        results = result.get("results", [])
        if results:
            lines = []
            for r in results:
                lines.append("[%s] %s (id:%s, score:%.2f)" % (
                    r.get("type", "?"), r.get("title", "?"),
                    r.get("id", "?")[:8], r.get("effective_activation", 0)))
                content = (r.get("content", "") or "")[:300]
                if content:
                    lines.append("  %s" % content)
                lines.append("")
            recall_context = "\n".join(lines)
    except Exception as e:
        recall_context = "Recall error: %s" % e

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Previous State\nFirst run.\n\n"
    content += "### Conversation (last %d exchanges)\n\n%s\n" % (len(exchanges), msg_text)
    if recall_context:
        content += "### Brain Context\n\n%s\n" % recall_context
    else:
        content += "### Brain Context\nNo recall data available.\n\n"
    return content


def _build_v3_content(conversation: dict, brain, counter: int = 5) -> str:
    """Build v3-style timeline with pre-attached recall + concept inventory."""
    exchanges = conversation["exchanges"]

    # Build timeline with simulated pre-attached recall per user turn
    timeline = ""
    turn_num = 0
    i = 0
    while i < len(exchanges):
        ex = exchanges[i]
        if ex["role"] == "user":
            turn_num += 1
            user_content = ex["content"][:800]

            # Simulate pre-attached recall for this user turn
            try:
                result = brain._brain.recall(query=user_content[:300], limit=5)
                results = result.get("results", [])
            except Exception:
                results = []

            timeline += "[TURN %d]\n" % turn_num
            timeline += "USER: \"%s\"\n" % user_content

            if results:
                timeline += "BRAIN SURFACED (%d nodes):\n" % len(results)
                for r in results:
                    timeline += "  [%s] %s (id:%s, score:%.2f)\n" % (
                        r.get("type", "?"), r.get("title", "?"),
                        r.get("id", "?")[:8], r.get("effective_activation", 0))
            else:
                timeline += "BRAIN SURFACED: nothing\n"

            # Include assistant response if next
            if i + 1 < len(exchanges) and exchanges[i + 1]["role"] == "assistant":
                timeline += "ASSISTANT: \"%s\"\n" % exchanges[i + 1]["content"][:800]
                i += 1

            timeline += "\n"
        i += 1

    # Concept inventory
    concept_inventory = ""
    try:
        rows = brain._brain.conn.execute(
            "SELECT id, title, content_summary, content FROM nodes "
            "WHERE type IN ('concept', 'vocabulary') AND archived=0"
        ).fetchall()
        if rows:
            lines = []
            for r in rows:
                summary = r[2] or (r[3] or "")[:80]
                lines.append("%s  %s — %s" % (r[0][:8], r[1], summary))
            concept_inventory = "\n".join(lines)
    except Exception:
        pass

    content = "## ENCODING RUN #%d\n\n" % counter
    content += "### Encoding Journal\nFirst run — no previous encoding in this session.\n\n"
    content += "### Conversation Timeline\n\n%s\n" % timeline
    if concept_inventory:
        content += "### Concept Inventory (%d nodes)\n\n%s\n\n" % (
            len(concept_inventory.split("\n")), concept_inventory)
    else:
        content += "### Concept Inventory\nNo concept nodes exist yet.\n\n"
    return content


# ── Run one variant ──

def run_variant(client, model: str, system_prompt: str, user_content: str,
                tools: list, brain: InstrumentedBrain,
                variant_name: str, verbose: bool = False) -> dict:
    """Run encoding agent with a specific prompt variant. Returns results dict."""

    # Reset captured actions
    brain.actions = []

    messages = [{"role": "user", "content": user_content}]

    t0 = time.time()
    total_input_tokens = 0
    total_output_tokens = 0

    response = client.messages.create(
        model=model, max_tokens=4096,
        system=system_prompt, messages=messages, tools=tools,
    )
    total_input_tokens += response.usage.input_tokens
    total_output_tokens += response.usage.output_tokens

    rounds = 0
    for rounds in range(8):
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
            if verbose:
                summary = tu.input.get("title", tu.input.get("query",
                    tu.input.get("node_id", "")))[:60]
                print("    [%s] %s" % (tu.name, summary))

        messages.append({"role": "assistant", "content": [
            {"type": b.type, **({"text": b.text} if b.type == "text" else
                                {"id": b.id, "name": b.name, "input": b.input})}
            for b in response.content]})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model, max_tokens=4096,
            system=system_prompt, messages=messages, tools=tools,
        )
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        rounds += 1

    elapsed = time.time() - t0

    # Extract final text
    final_text = "".join(b.text for b in response.content if b.type == "text")

    # Analyze actions
    actions = brain.actions
    creates = [a for a in actions if a.tool == "remember"]
    revises = [a for a in actions if a.tool == "revise"]
    connects = [a for a in actions if a.tool == "connect"]
    recalls = [a for a in actions if a.tool == "recall"]
    finds = [a for a in actions if a.tool == "find_node_by_title"]
    gets = [a for a in actions if a.tool == "get_node"]
    specialized = [a for a in actions if a.tool in (
        "remember_lesson", "remember_mechanism", "remember_mental_model",
        "remember_impact", "remember_convention")]
    divergences = [a for a in actions if a.tool == "record_divergence"]
    vocabs = [a for a in actions if a.tool == "learn_vocabulary"]

    # Measure content richness
    content_lengths = []
    for a in creates + specialized:
        c = a.args.get("content", a.args.get("what_happened", ""))
        content_lengths.append(len(c))

    # Check for concept-type nodes
    concept_creates = [a for a in creates if a.args.get("type") == "concept"]

    # Check for situation field usage
    with_situation = [a for a in creates if a.args.get("situation")]

    # Check for their_raw_quote usage
    with_quote = [a for a in creates + revises if a.args.get("their_raw_quote")]

    # Collect unique node types created
    created_types = list(set(a.args.get("type", "") for a in creates + specialized if a.args.get("type")))

    # Collect open/additional fields (beyond core: type, title, content, keywords)
    core_fields = {"type", "title", "content", "keywords", "locked", "confidence"}
    open_fields_used = set()
    for a in creates + revises + specialized:
        for k in a.args:
            if k not in core_fields and k not in ("node_id", "reason", "query", "limit",
                "source_id", "target_id", "relation", "weight", "title_query", "threshold",
                "term", "maps_to", "context", "claude_assumed", "reality", "underlying_pattern",
                "severity", "what_happened", "root_cause", "fix", "preventive_principle",
                "steps", "data_flow", "model_description", "applies_to",
                "pattern", "anti_pattern", "if_changed", "must_check", "because"):
                if a.args[k]:  # only count if non-empty
                    open_fields_used.add(k)

    # Check for journal in final text
    has_journal = any(marker in final_text for marker in ["ENCODED:", "SKIPPED:", "WATCHING:"])

    # Cost estimate (Sonnet 4: $3/MTok input, $15/MTok output)
    cost_input = total_input_tokens * 3.0 / 1_000_000
    cost_output = total_output_tokens * 15.0 / 1_000_000
    cost_total = cost_input + cost_output

    return {
        "variant": variant_name,
        "rounds": rounds + 1,
        "elapsed_s": round(elapsed, 1),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "cost_usd": round(cost_total, 4),
        "total_actions": len(actions),
        "creates": len(creates),
        "revises": len(revises),
        "connects": len(connects),
        "recalls": len(recalls),
        "find_by_title": len(finds),
        "get_node": len(gets),
        "specialized_tools": len(specialized),
        "divergences": len(divergences),
        "vocabs": len(vocabs),
        "concept_creates": len(concept_creates),
        "with_situation": len(with_situation),
        "with_quote": len(with_quote),
        "avg_content_length": round(sum(content_lengths) / max(len(content_lengths), 1)),
        "created_types": created_types,
        "open_fields_used": sorted(open_fields_used),
        "has_journal": has_journal,
        "final_text_preview": final_text[:800],
        "action_log": [
            {"tool": a.tool,
             "title": a.args.get("title", a.args.get("query", a.args.get("node_id", "")))[:60],
             "type": a.args.get("type", ""),
             "error": a.error}
            for a in actions
        ],
    }


# ── Main comparison ──

def run_comparison(corpus_path: str = None, live_brain: bool = False,
                   model: str = "claude-sonnet-4-6", verbose: bool = True) -> dict:
    """Run v2 vs v3 comparison on one or more conversations."""

    client = anthropic.Anthropic()

    # Load conversation(s)
    if corpus_path:
        with open(corpus_path) as f:
            conversations = [json.load(f)]
    else:
        conversations = load_corpus()
    if not conversations:
        print("No conversations found.")
        return {}

    # Setup brain
    if live_brain:
        db_dir = os.environ.get("BRAIN_DB_DIR",
            os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
        db_path = os.path.join(db_dir, "brain.db")
        work_dir = None
    else:
        # Copy fixture brain
        fixture = ROOT / "eval" / "fixtures" / "brain_eval_copy.db"
        if not fixture.exists():
            # Fall back to live brain copy
            db_dir = os.environ.get("BRAIN_DB_DIR",
                os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
            fixture = Path(os.path.join(db_dir, "brain.db"))
        work_dir = tempfile.mkdtemp(prefix="brain_v3_compare_")
        db_path = os.path.join(work_dir, "brain.db")
        shutil.copy2(str(fixture), db_path)
        # Copy logs db too
        logs_src = str(fixture).replace("brain.db", "brain_logs.db")
        if os.path.exists(logs_src):
            shutil.copy2(logs_src, os.path.join(work_dir, "brain_logs.db"))

    from servers.brain import Brain

    # Load prompts
    v2_prompt = _load_v2_prompt()
    v3_prompt = _load_v3_prompt()

    if verbose:
        print("=" * 70)
        print("ENCODING AGENT v2 vs v3 COMPARISON")
        print("=" * 70)
        print("Model: %s" % model)
        print("Conversations: %d" % len(conversations))
        print("Brain: %s" % ("live" if live_brain else db_path))
        print("v2 prompt: %d chars, v3 prompt: %d chars" % (len(v2_prompt), len(v3_prompt)))
        print("v2 tools: %d, v3 tools: %d" % (len(V2_TOOLS), len(V3_TOOLS)))
        print()

    all_results = []

    for conv in conversations:
        if verbose:
            print("-" * 50)
            print("Conversation: %s (%s)" % (conv["id"], conv.get("category", "?")))
            print("-" * 50)

        # ── Run V2 ──
        if verbose:
            print("\n  [V2] Running with v2 prompt + 13 tools...")

        brain_v2 = Brain(db_path=db_path)
        instrumented_v2 = InstrumentedBrain(brain_v2)
        user_content_v2 = _build_v2_content(conv, instrumented_v2)

        if verbose:
            print("  [V2] User content: %d chars" % len(user_content_v2))

        result_v2 = run_variant(
            client, model, v2_prompt, user_content_v2,
            V2_TOOLS, instrumented_v2, "v2", verbose=verbose)

        brain_v2.close()

        # ── Run V3 ──
        if verbose:
            print("\n  [V3] Running with v3 prompt + 8 tools...")

        brain_v3 = Brain(db_path=db_path)
        instrumented_v3 = InstrumentedBrain(brain_v3)
        user_content_v3 = _build_v3_content(conv, instrumented_v3)

        if verbose:
            print("  [V3] User content: %d chars" % len(user_content_v3))

        result_v3 = run_variant(
            client, model, v3_prompt, user_content_v3,
            V3_TOOLS, instrumented_v3, "v3", verbose=verbose)

        brain_v3.close()

        # ── Compare ──
        comparison = {
            "conversation_id": conv["id"],
            "category": conv.get("category", ""),
            "v2": result_v2,
            "v3": result_v3,
            "delta": {
                "rounds": result_v3["rounds"] - result_v2["rounds"],
                "creates": result_v3["creates"] - result_v2["creates"],
                "revises": result_v3["revises"] - result_v2["revises"],
                "connects": result_v3["connects"] - result_v2["connects"],
                "avg_content_length": result_v3["avg_content_length"] - result_v2["avg_content_length"],
                "concept_creates": result_v3["concept_creates"],
                "with_situation": result_v3["with_situation"] - result_v2["with_situation"],
                "specialized_removed": result_v2["specialized_tools"],
                "journal_present": result_v3["has_journal"],
            }
        }
        all_results.append(comparison)

        if verbose:
            print("\n  COMPARISON:")
            print("  %-25s  %10s  %10s  %10s" % ("Metric", "v2", "v3", "delta"))
            print("  " + "-" * 60)
            for key in ["rounds", "elapsed_s", "input_tokens", "output_tokens",
                        "total_tokens", "cost_usd",
                        "total_actions", "creates", "revises", "connects",
                        "recalls", "find_by_title", "get_node",
                        "specialized_tools", "divergences", "vocabs",
                        "concept_creates", "with_situation", "with_quote",
                        "avg_content_length"]:
                v2_val = result_v2[key]
                v3_val = result_v3[key]
                delta = v3_val - v2_val if isinstance(v2_val, (int, float)) else ""
                sign = "+" if isinstance(delta, (int, float)) and delta > 0 else ""
                fmt = "%.4f" if key == "cost_usd" else "%s"
                print("  %-25s  %10s  %10s  %10s" % (key,
                      fmt % v2_val if isinstance(v2_val, float) and key == "cost_usd" else v2_val,
                      fmt % v3_val if isinstance(v3_val, float) and key == "cost_usd" else v3_val,
                      "%s%s" % (sign, round(delta, 4) if isinstance(delta, float) else delta) if delta != "" else ""))

            print("\n  v2 types created: %s" % result_v2.get("created_types", []))
            print("  v3 types created: %s" % result_v3.get("created_types", []))
            print("  v2 open fields: %s" % result_v2.get("open_fields_used", []))
            print("  v3 open fields: %s" % result_v3.get("open_fields_used", []))
            print("  v3 journal present: %s" % result_v3["has_journal"])
            print("\n  v3 final text (preview):")
            print("    %s" % result_v3["final_text_preview"][:300])

    # Save results
    results_dir = ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / ("v3_compare_%s.json" % ts)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    if verbose:
        print("\n\nResults saved to: %s" % results_path)

    # Cleanup
    if work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return {"results": all_results, "path": str(results_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoding Agent v2 vs v3 Comparison")
    parser.add_argument("--corpus", help="Path to conversation JSON file")
    parser.add_argument("--live-brain", action="store_true", help="Use live brain instead of fixture")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to use")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    run_comparison(
        corpus_path=args.corpus,
        live_brain=args.live_brain,
        model=args.model,
        verbose=not args.quiet,
    )
