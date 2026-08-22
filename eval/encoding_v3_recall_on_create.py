#!/usr/bin/env python3
"""Encoding Agent v3 vs v3+recall-on-create — quality-focused comparison.

Tests the hypothesis: if remember() returns related nodes in its response,
does the agent make better connections and avoid bad ones?

Measures:
- Connection quality: semantic similarity between connected nodes
- Node quality: content richness, situation presence, type diversity
- Behavioral: does the agent connect immediately or batch?
- Efficiency: rounds, tokens, cost

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python3 eval/encoding_v3_recall_on_create.py --corpus eval/corpus/conv_004_art_design_extended.json
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

# Load .env
_env_path = ROOT / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ[_k.strip()] = _v.strip()

import anthropic
from eval.capabilities.base import InstrumentedBrain, CapturedAction, dispatch_tool


# ── Prompt + tools ──

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


def _load_v3_plus_prompt():
    """v3 prompt with recall-on-create + batch + efficiency."""
    prompt = _load_v3_prompt()
    prompt += """

## Recall on Create

When you call `remember()`, the response includes up to 5 related nodes (full content). Use these to `connect()` immediately in the next round. Read the content to judge whether connections are meaningful.

## Batch Everything

Call MULTIPLE `remember()` and `connect()` in the SAME round. Sonnet supports parallel tool calls. Don't spread 5 creates across 5 rounds — do all 5 in one round, then all connections in the next.

## You Run Every 5 Messages

You run every 5 messages — this isn't the only chance. Act on what's clear, skip what's ambiguous.

## Speed

The conversation timeline IS your recall context. Do NOT recall topics already surfaced there.

Target: **2-3 rounds.**
- Round 1: read timeline. Call ALL `remember()` + `find_node_by_title()` + `revise()` calls at once.
- Round 2: use related_nodes from remember() responses to `connect()` everything. Journal + DONE.
"""
    return prompt


V3_TOOL_NAMES = {
    'recall', 'find_node_by_title', 'get_node',
    'remember', 'revise', 'connect',
    'record_divergence', 'learn_vocabulary',
}

V3_PLUS_TOOL_NAMES = {
    'recall', 'find_node_by_title', 'get_node',
    'remember_batch', 'revise',
    'record_divergence', 'learn_vocabulary',
}


def _build_tools(tool_names=None):
    from eval.capabilities.base import _build_capability_tools
    all_tools = _build_capability_tools()
    names = tool_names or V3_TOOL_NAMES
    # Add tools from brain_mcp that aren't in capability_tools
    from servers import brain_mcp
    existing_names = {t["name"] for t in all_tools}
    for t in brain_mcp.TOOLS:
        if t["name"] in names and t["name"] not in existing_names:
            all_tools.append({"name": t["name"], "description": t["description"],
                              "input_schema": t["inputSchema"]})
    return [t for t in all_tools if t["name"] in names]


# ── Dispatch with recall-on-create ──

def _get_related_for_node(brain, node_id, exclude_ids=None):
    """Get related nodes for a given node ID. Returns list of related node dicts."""
    exclude = set(exclude_ids or [])
    exclude.add(node_id)
    try:
        node = brain._brain.get_node(node_id)
        if not node:
            return []
        query = "%s %s" % (node.get("title", ""), (node.get("content", "") or "")[:200])
        recall_result = brain._brain.recall(query=query, limit=6)
        related = [r for r in recall_result.get("results", []) if r.get("id") not in exclude][:5]
        return [{
            "id": r.get("id", ""), "type": r.get("type", ""),
            "title": r.get("title", ""),
            "content": (r.get("content", "") or "")[:500],
            "confidence": r.get("confidence", 0),
            "score": round(r.get("effective_activation", 0), 3),
        } for r in related]
    except Exception:
        return []


def dispatch_with_recall(brain: InstrumentedBrain, tool_name: str, tool_input: dict,
                          return_related: bool = False) -> str:
    """Dispatch tool call. If return_related, append related nodes to remember/encode_cluster results."""

    # Handle remember_batch
    if tool_name == "remember_batch":
        try:
            result = brain._brain.remember_batch(
                nodes=tool_input.get("nodes", []),
                connect_to=tool_input.get("connect_to"),
                auto_connect=tool_input.get("auto_connect", True))
            brain.actions.append(CapturedAction(
                tool="remember_batch", args=tool_input, result=result, timestamp=time.time()))
            return json.dumps({"ok": True, "result": result}, default=str)
        except Exception as e:
            brain.actions.append(CapturedAction(
                tool="remember_batch", args=tool_input, error=str(e), timestamp=time.time()))
            return json.dumps({"ok": False, "error": str(e)})

    # Standard dispatch for other tools
    result_text = dispatch_tool(brain, tool_name, tool_input)

    if not return_related or tool_name != "remember":
        return result_text

    # Append related nodes to remember() result
    try:
        result = json.loads(result_text)
        node_id = result.get("id", "")
        if not node_id or not result.get("ok"):
            return result_text
        result["related_nodes"] = _get_related_for_node(brain, node_id)
        return json.dumps(result)
    except Exception:
        return result_text


# ── Content builders ──

def _build_v3_content(conversation: dict, brain) -> str:
    """Build v3-style timeline with pre-attached recall."""
    exchanges = conversation["exchanges"]
    timeline = ""
    turn_num = 0
    i = 0
    while i < len(exchanges):
        ex = exchanges[i]
        if ex["role"] == "user":
            turn_num += 1
            user_content = ex["content"][:800]
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

            if i + 1 < len(exchanges) and exchanges[i + 1]["role"] == "assistant":
                timeline += "ASSISTANT: \"%s\"\n" % exchanges[i + 1]["content"][:800]
                i += 1
            timeline += "\n"
        i += 1

    content = "## ENCODING RUN #5\n\n"
    content += "### Encoding Journal\nFirst run — no previous encoding in this session.\n\n"
    content += "### Conversation Timeline\n\n%s\n" % timeline
    content += "### Concept Inventory\nNo concept nodes exist yet.\n\n"
    return content


# ── Run one variant ──

def run_variant(client, model: str, system_prompt: str, user_content: str,
                tools: list, brain: InstrumentedBrain,
                variant_name: str, return_related: bool = False,
                verbose: bool = False) -> dict:
    """Run encoding agent. Returns comprehensive results."""

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
    max_rounds = 5 if return_related else 12
    for rounds in range(max_rounds):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            result_text = dispatch_with_recall(brain, tu.name, tu.input,
                                                return_related=return_related)
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
    final_text = "".join(b.text for b in response.content if b.type == "text")

    # ── Analyze actions ──
    actions = brain.actions
    creates = [a for a in actions if a.tool == "remember"]
    clusters = [a for a in actions if a.tool == "encode_cluster"]
    revises = [a for a in actions if a.tool == "revise"]
    connects = [a for a in actions if a.tool == "connect"]
    recalls = [a for a in actions if a.tool == "recall"]
    finds = [a for a in actions if a.tool == "find_node_by_title"]
    gets = [a for a in actions if a.tool == "get_node"]
    divergences = [a for a in actions if a.tool == "record_divergence"]

    # Count nodes from encode_cluster
    cluster_node_count = 0
    cluster_connect_count = 0
    cluster_node_specs = []
    for c in clusters:
        if c.result and isinstance(c.result, dict):
            cluster_node_count += c.result.get("nodes_created", 0)
            cluster_connect_count += c.result.get("connections_created", 0)
        for node_spec in c.args.get("nodes", []):
            cluster_node_specs.append(node_spec)

    # Content metrics — from both remember() and encode_cluster node specs
    content_lengths = [len(a.args.get("content", "")) for a in creates if a.args.get("content")]
    content_lengths += [len(n.get("content", "")) for n in cluster_node_specs if n.get("content")]

    all_node_specs = [a.args for a in creates] + cluster_node_specs
    concept_creates = [n for n in all_node_specs if n.get("type") == "concept"]
    with_situation = [n for n in all_node_specs if n.get("situation")]
    with_quote = [n for n in all_node_specs + [a.args for a in revises] if n.get("their_raw_quote")]
    created_types = sorted(set(n.get("type", "") for n in all_node_specs if n.get("type")))

    # Open fields
    core_fields = {"type", "title", "content", "keywords", "locked", "confidence",
                   "node_id", "reason", "query", "limit", "source_id", "target_id",
                   "relation", "weight", "title_query", "threshold",
                   "term", "maps_to", "context", "claude_assumed", "reality",
                   "underlying_pattern", "severity"}
    open_fields = set()
    for a in creates + revises:
        for k, v in a.args.items():
            if k not in core_fields and v:
                open_fields.add(k)

    # ── Connection quality analysis ──
    connection_details = []
    created_node_ids = set()
    for a in creates:
        if a.result and isinstance(a.result, dict):
            nid = a.result.get("id", "")
            if nid:
                created_node_ids.add(nid)

    for a in connects:
        src = a.args.get("source_id", "")
        tgt = a.args.get("target_id", "")
        relation = a.args.get("relation", "related_to")

        # Get both nodes to measure similarity
        src_node = brain._brain.get_node(src) if src else None
        tgt_node = brain._brain.get_node(tgt) if tgt else None

        detail = {
            "source_id": src[:8],
            "target_id": tgt[:8],
            "relation": relation,
            "source_title": src_node.get("title", "?")[:50] if src_node else "?",
            "target_title": tgt_node.get("title", "?")[:50] if tgt_node else "?",
            "source_type": src_node.get("type", "?") if src_node else "?",
            "target_type": tgt_node.get("type", "?") if tgt_node else "?",
            "new_to_existing": src in created_node_ids and tgt not in created_node_ids,
            "new_to_new": src in created_node_ids and tgt in created_node_ids,
            "error": a.error or "",
        }

        # Compute semantic similarity if both nodes have content
        if src_node and tgt_node:
            try:
                from servers import embedder
                src_text = "%s %s" % (src_node.get("title", ""), (src_node.get("content", "") or "")[:200])
                tgt_text = "%s %s" % (tgt_node.get("title", ""), (tgt_node.get("content", "") or "")[:200])
                src_emb = embedder.embed(src_text)
                tgt_emb = embedder.embed(tgt_text)
                sim = embedder.cosine_similarity(src_emb, tgt_emb)
                detail["cosine_similarity"] = round(sim, 3)
            except Exception:
                detail["cosine_similarity"] = None
        else:
            detail["cosine_similarity"] = None

        connection_details.append(detail)

    # ── Node quality details ──
    node_details = []
    for a in creates:
        node_details.append({
            "title": a.args.get("title", "")[:60],
            "type": a.args.get("type", ""),
            "content_length": len(a.args.get("content", "")),
            "has_situation": bool(a.args.get("situation")),
            "has_keywords": bool(a.args.get("keywords")),
            "has_quote": bool(a.args.get("their_raw_quote")),
            "situation": (a.args.get("situation", "") or "")[:80],
            "error": a.error or "",
        })

    # Journal
    has_journal = any(m in final_text for m in ["ENCODED:", "SKIPPED:", "WATCHING:"])

    # Cost
    cost = (total_input_tokens * 3.0 + total_output_tokens * 15.0) / 1_000_000

    # ── Sequence analysis: did connects follow creates? ──
    create_then_connect = 0
    for i, a in enumerate(actions):
        if a.tool == "remember" and i + 1 < len(actions):
            # Check if next action (or within next 3) is a connect
            for j in range(i + 1, min(i + 4, len(actions))):
                if actions[j].tool == "connect":
                    create_then_connect += 1
                    break

    return {
        "variant": variant_name,
        "rounds": rounds + 1,
        "elapsed_s": round(elapsed, 1),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost_usd": round(cost, 4),
        "creates": len(creates),
        "revises": len(revises),
        "connects": len(connects),
        "recalls": len(recalls),
        "find_by_title": len(finds),
        "get_node": len(gets),
        "divergences": len(divergences),
        "concept_creates": len(concept_creates),
        "with_situation": len(with_situation),
        "with_quote": len(with_quote),
        "avg_content_length": round(sum(content_lengths) / max(len(content_lengths), 1)),
        "created_types": created_types,
        "open_fields": sorted(open_fields),
        "has_journal": has_journal,
        "create_then_connect": create_then_connect,
        "connection_details": connection_details,
        "node_details": node_details,
        "avg_connection_similarity": round(
            sum(c["cosine_similarity"] for c in connection_details if c["cosine_similarity"] is not None) /
            max(len([c for c in connection_details if c["cosine_similarity"] is not None]), 1), 3),
        "final_text": final_text[:1000],
        "action_log": [
            {"tool": a.tool,
             "title": a.args.get("title", a.args.get("query", a.args.get("node_id", "")))[:60],
             "type": a.args.get("type", ""),
             "error": a.error or ""}
            for a in actions
        ],
    }


# ── Pretty print ──

def print_three_way(v2_result, v3_result, v3p_result):
    v2, v3, v3p = v2_result, v3_result, v3p_result

    print()
    print('┌──────────────────────────────────────────────────────────────────────────────┐')
    print('│       ENCODING v2 vs v3 vs v3+recall-on-create — 3-Way Comparison           │')
    print('└──────────────────────────────────────────────────────────────────────────────┘')
    print()

    def row(label, v2v, v3v, v3pv, fmt='%d'):
        print('│ %-28s │ %8s │ %8s │ %8s │' % (
            label, fmt % v2v, fmt % v3v, fmt % v3pv))

    def header(label):
        print('├──────────────────────────────┼──────────┼──────────┼──────────┤')
        print('│ %-28s │          │          │          │' % label)
        print('├──────────────────────────────┼──────────┼──────────┼──────────┤')

    print('┌──────────────────────────────┬──────────┬──────────┬──────────┐')
    print('│ Metric                       │    v2    │    v3    │   v3+    │')
    print('├──────────────────────────────┼──────────┼──────────┼──────────┤')

    header('EFFICIENCY')
    row('Rounds', v2['rounds'], v3['rounds'], v3p['rounds'])
    row('Time (s)', v2['elapsed_s'], v3['elapsed_s'], v3p['elapsed_s'], '%.1f')
    row('Input tokens', v2['input_tokens'], v3['input_tokens'], v3p['input_tokens'])
    row('Output tokens', v2['output_tokens'], v3['output_tokens'], v3p['output_tokens'])
    row('Cost (USD)', v2['cost_usd'], v3['cost_usd'], v3p['cost_usd'], '%.4f')

    header('ENCODING ACTIONS')
    row('Creates', v2['creates'], v3['creates'], v3p['creates'])
    row('Revises', v2['revises'], v3['revises'], v3p['revises'])
    row('Connects', v2['connects'], v3['connects'], v3p['connects'])
    row('Create→Connect pairs', v2['create_then_connect'], v3['create_then_connect'], v3p['create_then_connect'])
    row('Concept creates', v2.get('concept_creates',0), v3.get('concept_creates',0), v3p.get('concept_creates',0))
    row('Divergences', v2['divergences'], v3['divergences'], v3p['divergences'])

    header('SEARCH BEHAVIOR')
    row('Recalls', v2['recalls'], v3['recalls'], v3p['recalls'])
    row('find_node_by_title', v2['find_by_title'], v3['find_by_title'], v3p['find_by_title'])
    row('get_node', v2['get_node'], v3['get_node'], v3p['get_node'])

    header('QUALITY')
    row('Avg content length', v2['avg_content_length'], v3['avg_content_length'], v3p['avg_content_length'])
    row('With situation', v2['with_situation'], v3['with_situation'], v3p['with_situation'])
    row('With quote', v2['with_quote'], v3['with_quote'], v3p['with_quote'])
    row('Avg conn similarity', int(v2['avg_connection_similarity']*1000),
        int(v3['avg_connection_similarity']*1000), int(v3p['avg_connection_similarity']*1000))
    row('Journal present', 1 if v2['has_journal'] else 0,
        1 if v3['has_journal'] else 0, 1 if v3p['has_journal'] else 0)

    print('└──────────────────────────────┴──────────┴──────────┴──────────┘')

    print()
    print('TYPES:  v2=%s' % v2['created_types'])
    print('        v3=%s' % v3['created_types'])
    print('        v3+=%s' % v3p['created_types'])
    print('FIELDS: v2=%s' % v2['open_fields'])
    print('        v3=%s' % v3['open_fields'])
    print('        v3+=%s' % v3p['open_fields'])

    for label, r in [('v2', v2), ('v3', v3), ('v3+', v3p)]:
        print()
        print('%s NODES (%d):' % (label, len(r['node_details'])))
        for n in r['node_details']:
            print('  [%-15s] %-42s %4dc sit=%s' % (
                n['type'], n['title'][:42], n['content_length'],
                'Y' if n['has_situation'] else 'N'))
        print('%s CONNECTIONS (%d):' % (label, len(r['connection_details'])))
        for c in r['connection_details']:
            sim = '%.3f' % c['cosine_similarity'] if c['cosine_similarity'] else 'N/A'
            tag = 'NEW→EXIST' if c['new_to_existing'] else ('NEW→NEW' if c['new_to_new'] else 'EX→EX')
            print('  %-28s → %-28s [%-10s] sim=%s %s' % (
                c['source_title'][:28], c['target_title'][:28], c['relation'], sim, tag))


def print_comparison(v3_result, v3p_result):
    print()
    print('┌─────────────────────────────────────────────────────────────────────────┐')
    print('│    ENCODING v3 vs v3+recall-on-create — Quality-Focused Comparison     │')
    print('└─────────────────────────────────────────────────────────────────────────┘')
    print()

    v3, v3p = v3_result, v3p_result

    def row(label, v3v, v3pv, higher_is_better=True, fmt='%d'):
        d = v3pv - v3v
        sign = '+' if d > 0 else ''
        if higher_is_better:
            w = 'v3+' if d > 0 else ('v3' if d < 0 else '  ')
        else:
            w = 'v3+' if d < 0 else ('v3' if d > 0 else '  ')
        print('│ %-28s │ %8s │ %8s │ %8s │  %-3s │' % (
            label, fmt % v3v, fmt % v3pv, ('%s' + fmt) % (sign, d), w))

    def header(label):
        print('├──────────────────────────────┼──────────┼──────────┼──────────┼──────┤')
        print('│ %-28s │          │          │          │      │' % label)
        print('├──────────────────────────────┼──────────┼──────────┼──────────┼──────┤')

    print('┌──────────────────────────────┬──────────┬──────────┬──────────┬──────┐')
    print('│ Metric                       │    v3    │   v3+    │   delta  │  win │')
    print('├──────────────────────────────┼──────────┼──────────┼──────────┼──────┤')

    header('EFFICIENCY')
    row('Rounds', v3['rounds'], v3p['rounds'], False)
    row('Time (s)', v3['elapsed_s'], v3p['elapsed_s'], False, '%.1f')
    row('Total tokens', v3['input_tokens']+v3['output_tokens'],
        v3p['input_tokens']+v3p['output_tokens'], False)
    row('Cost (USD)', v3['cost_usd'], v3p['cost_usd'], False, '%.4f')

    header('ENCODING ACTIONS')
    row('Creates', v3['creates'], v3p['creates'], True)
    row('Revises', v3['revises'], v3p['revises'], True)
    row('Connects', v3['connects'], v3p['connects'], True)
    row('Create→Connect pairs', v3['create_then_connect'], v3p['create_then_connect'], True)
    row('Concept creates', v3['concept_creates'], v3p['concept_creates'], True)

    header('SEARCH BEHAVIOR')
    row('Recalls', v3['recalls'], v3p['recalls'], False)
    row('find_node_by_title', v3['find_by_title'], v3p['find_by_title'], True)
    row('get_node', v3['get_node'], v3p['get_node'], True)

    header('QUALITY')
    row('Avg content length', v3['avg_content_length'], v3p['avg_content_length'], True)
    row('With situation', v3['with_situation'], v3p['with_situation'], True)
    row('With quote', v3['with_quote'], v3p['with_quote'], True)
    row('Avg conn similarity', int(v3['avg_connection_similarity']*1000),
        int(v3p['avg_connection_similarity']*1000), True)
    row('Journal present', 1 if v3['has_journal'] else 0, 1 if v3p['has_journal'] else 0, True)

    print('└──────────────────────────────┴──────────┴──────────┴──────────┴──────┘')

    print()
    print('TYPES:  v3=%s' % v3['created_types'])
    print('        v3+=%s' % v3p['created_types'])
    print('FIELDS: v3=%s' % v3['open_fields'])
    print('        v3+=%s' % v3p['open_fields'])

    # Node details
    print()
    print('CREATED NODES:')
    print('  v3:')
    for n in v3['node_details']:
        print('    [%-15s] %-45s %4d chars  sit=%s' % (
            n['type'], n['title'][:45], n['content_length'],
            'Y' if n['has_situation'] else 'N'))
    print('  v3+:')
    for n in v3p['node_details']:
        print('    [%-15s] %-45s %4d chars  sit=%s' % (
            n['type'], n['title'][:45], n['content_length'],
            'Y' if n['has_situation'] else 'N'))

    # Connection details
    print()
    print('CONNECTIONS:')
    print('  v3:')
    if v3['connection_details']:
        for c in v3['connection_details']:
            sim_str = '%.3f' % c['cosine_similarity'] if c['cosine_similarity'] else 'N/A'
            print('    %s → %s  [%s]  sim=%s  %s' % (
                c['source_title'][:30], c['target_title'][:30],
                c['relation'], sim_str,
                'NEW→EXIST' if c['new_to_existing'] else ('NEW→NEW' if c['new_to_new'] else 'EXIST→EXIST')))
    else:
        print('    (none)')
    print('  v3+:')
    if v3p['connection_details']:
        for c in v3p['connection_details']:
            sim_str = '%.3f' % c['cosine_similarity'] if c['cosine_similarity'] else 'N/A'
            print('    %s → %s  [%s]  sim=%s  %s' % (
                c['source_title'][:30], c['target_title'][:30],
                c['relation'], sim_str,
                'NEW→EXIST' if c['new_to_existing'] else ('NEW→NEW' if c['new_to_new'] else 'EXIST→EXIST')))
    else:
        print('    (none)')


# ── Main ──

def _build_v2_content(conversation: dict, brain) -> str:
    """Build v2-style flat user content."""
    exchanges = conversation["exchanges"]
    msg_text = ""
    for ex in exchanges:
        msg_text += "[%s]: %s\n\n" % (ex["role"].upper(), ex["content"][:600])

    user_msgs = [ex["content"] for ex in exchanges if ex["role"] == "user"]
    recall_query = " ".join(msg[:200] for msg in user_msgs[-3:])
    recall_context = ""
    try:
        result = brain._brain.recall(query=recall_query, limit=5)
        for r in result.get("results", []):
            recall_context += "[%s] %s (id:%s)\n  %s\n\n" % (
                r.get("type","?"), r.get("title","?"), r.get("id","?")[:8],
                (r.get("content","") or "")[:300])
    except Exception:
        pass

    content = "## ENCODING RUN #5\n\n### Previous State\nFirst run.\n\n"
    content += "### Conversation (last %d exchanges)\n\n%s\n" % (len(exchanges), msg_text)
    content += "### Brain Context\n\n%s\n" % (recall_context or "No recall data.")
    return content


def _build_v2_tools():
    from eval.capabilities.base import _build_capability_tools
    return _build_capability_tools()  # All 13 tools


def run(corpus_path: str, model: str = "claude-sonnet-4-6", verbose: bool = True, db_override: str = None):
    client = anthropic.Anthropic()

    with open(corpus_path) as f:
        conversation = json.load(f)

    if db_override:
        db_path = db_override
    else:
        db_dir = os.environ.get("BRAIN_DB_DIR",
            os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
        db_path = os.path.join(db_dir, "brain.db")

    v3_tools = _build_tools(V3_TOOL_NAMES)
    v3p_tools = _build_tools(V3_PLUS_TOOL_NAMES)
    v2_tools = _build_v2_tools()
    v2_prompt = _load_v2_prompt()
    v3_prompt = _load_v3_prompt()
    v3p_prompt = _load_v3_plus_prompt()

    from servers.brain import Brain

    if verbose:
        print("Conversation: %s (%d exchanges)" % (conversation["id"], len(conversation["exchanges"])))
        print("Model: %s" % model)
        print()

    # ── v2 baseline ──
    if verbose:
        print("[v2] Running v2 baseline...")
    brain_v2 = Brain(db_path=db_path)
    inst_v2 = InstrumentedBrain(brain_v2)
    content_v2 = _build_v2_content(conversation, inst_v2)
    result_v2 = run_variant(client, model, v2_prompt, content_v2, v2_tools, inst_v2,
                             "v2", return_related=False, verbose=verbose)
    brain_v2.close()

    # ── v3 ──
    if verbose:
        print("\n[v3] Running v3...")
    brain_v3 = Brain(db_path=db_path)
    inst_v3 = InstrumentedBrain(brain_v3)
    content_v3 = _build_v3_content(conversation, inst_v3)
    result_v3 = run_variant(client, model, v3_prompt, content_v3, v3_tools, inst_v3,
                             "v3", return_related=False, verbose=verbose)
    brain_v3.close()

    # ── v3+ with recall-on-create ──
    if verbose:
        print("\n[v3+] Running v3 + recall-on-create...")
    brain_v3p = Brain(db_path=db_path)
    inst_v3p = InstrumentedBrain(brain_v3p)
    content_v3p = _build_v3_content(conversation, inst_v3p)
    result_v3p = run_variant(client, model, v3p_prompt, content_v3p, v3p_tools, inst_v3p,
                              "v3+batch", return_related=True, verbose=verbose)
    brain_v3p.close()

    # ── Print 3-way comparison ──
    print_three_way(result_v2, result_v3, result_v3p)

    # Save
    results_dir = ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = results_dir / ("v3_recall_on_create_%s.json" % ts)
    with open(path, "w") as f:
        json.dump({"v2": result_v2, "v3": result_v3, "v3_plus": result_v3p}, f, indent=2, default=str)
    print("\nResults saved to: %s" % path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Conversation JSON file")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--db", help="Path to brain.db (default: copy from BRAIN_DB_DIR)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(args.corpus, args.model, not args.quiet, db_override=args.db)
