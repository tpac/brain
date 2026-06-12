#!/usr/bin/env python3
"""Lens interview for an MCP tool definition — Sonnet reads, we map ambiguity.

Companion to eval/mcp_batch_probe.py (behavioral signal). This is the
DIAGNOSTIC half: send the same tool definition to N clean Sonnets, each
interrogating one aspect (Tom's multi-lens technique from
eval/encoder_prompt_probe.py, adapted to tool definitions). Output is an
ambiguity map of how the CONSUMER model reads the definition — not a metric
to optimize, an input to drafting variants.

Isolation: API-only. No brain, no daemon, no DB.

USE
---
    ./dev python3 eval/mcp_tool_interview.py                # live brain_batch
    ./dev python3 eval/mcp_tool_interview.py --tool remember_batch
    ./dev python3 eval/mcp_tool_interview.py --variant eval/mcp_variants/v2.json

Output: eval/mcp_variants/interview_{tag}.md
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = 'claude-sonnet-4-6'   # the consumer model — S1/S2 encoders
MAX_TOKENS = 1500

SYSTEM = (
    "You are an AI agent that uses tools via JSON tool definitions. You are "
    "being interviewed about ONE tool definition. Answer only from the "
    "definition text given — not from prior knowledge of similar tools. Be "
    "direct and concrete; if something is unclear in the definition, say so "
    "plainly."
)

LENSES = [
    {
        'id': 'contract_recall',
        'title': 'Contract recall — required fields per op',
        'prompt': (
            "From this tool definition ALONE: list every operation the tool "
            "accepts. For each operation, name its REQUIRED fields and what "
            "each required field means. If the definition does not state a "
            "requirement explicitly, say 'not stated' — do not guess."),
    },
    {
        'id': 'usage_writeout',
        'title': 'Usage write-out — the reason/reasoning probe',
        'prompt': (
            "Using only this definition, write the exact `operations` array "
            "you would emit to: update the stored 'why this node was encoded' "
            "rationale of node ab12cd34 to 'derived from the v29 trace "
            "audit', given that the motivation for making this change is "
            "'rationale was stale after the schema migration'. Output the "
            "JSON array and one sentence on any field-naming doubt you had."),
    },
    {
        'id': 'traps',
        'title': 'Trap inventory — what a hurried agent gets wrong',
        'prompt': (
            "You are reviewing this definition on behalf of a teammate model "
            "that works fast and makes plausible-looking mistakes. List the "
            "traps: places where a hurried agent would emit something invalid "
            "or subtly wrong. Rank by likelihood. For each trap, quote the "
            "definition text that should prevent it — or note that nothing "
            "in the definition prevents it."),
    },
    {
        'id': 'emphasis',
        'title': 'Emphasis audit — word budget vs importance',
        'prompt': (
            "What does this definition spend the most words on? What is "
            "underweighted relative to its importance for emitting correct "
            "calls? Name the top 3 over-weighted and top 3 under-weighted "
            "topics, with a one-line justification each."),
    },
    {
        'id': 'structure_split',
        'title': 'Prose vs schema — what belongs where',
        'prompt': (
            "Split this definition's prose into two buckets: (A) statements "
            "that duplicate what the JSON schema already encodes or COULD "
            "encode structurally (required fields, enums, per-op shapes), and "
            "(B) judgment guidance that only prose can carry. Quote 3-5 "
            "examples per bucket. Where the schema could carry something the "
            "prose currently does, say what the schema change would be."),
    },
    {
        'id': 'blindspots',
        'title': 'Blind spots — unanswered usage questions',
        'prompt': (
            "What questions about using this tool CORRECTLY does the "
            "definition fail to answer? Think: field semantics, what happens "
            "on partial failure, interactions between operations in one "
            "call, defaults. List each unanswered question and why it "
            "matters for correct usage."),
    },
]


def load_tool_def(tool_name='brain_batch', variant_path=None):
    if variant_path:
        with open(variant_path) as f:
            tool = json.load(f)
        tag = Path(variant_path).stem
    else:
        from servers import brain_mcp
        tool = next(t for t in brain_mcp.TOOLS if t['name'] == tool_name)
        tag = '%s_v0_live' % tool_name
    return tool, tag


def run_lens(client, lens, tool_text):
    resp = client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
        messages=[{'role': 'user', 'content':
                   lens['prompt'] + '\n\nTOOL DEFINITION:\n```json\n' + tool_text + '\n```'}],
    )
    return ''.join(b.text for b in resp.content if b.type == 'text')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tool', default='brain_batch')
    ap.add_argument('--variant', help='JSON file with a full tool def')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    tool, tag = load_tool_def(args.tool, args.variant)
    tool_text = json.dumps(tool, indent=2)

    out = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_lens, client, lens, tool_text): lens for lens in LENSES}
        for fut in as_completed(futs):
            lens = futs[fut]
            try:
                out[lens['id']] = fut.result()
            except Exception as e:
                out[lens['id']] = 'ERROR: %s' % str(e)[:200]
            print('lens done: %s' % lens['id'])

    out_dir = ROOT / 'eval' / 'mcp_variants'
    out_dir.mkdir(exist_ok=True)
    lines = ['# Tool-definition interview — %s' % tag,
             '', 'model=%s — each lens is an independent stateless call' % MODEL, '']
    for lens in LENSES:
        lines += ['## %s' % lens['title'], '', out.get(lens['id'], '(missing)'), '']
    path = out_dir / ('interview_%s.md' % tag)
    path.write_text('\n'.join(lines))
    print('\nwrote %s' % path)


if __name__ == '__main__':
    main()
