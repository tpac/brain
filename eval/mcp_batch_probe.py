#!/usr/bin/env python3
"""Generation-shape probes for the brain_batch MCP tool definition.

The optimization signal for tool-description work: put the CONSUMER model
(Sonnet — S1/S2 encoders) in a minimal encoder context with a candidate
(schema + description), bait one known failure mode per scenario, and
mechanically grade the tool call it emits. Pass rates per dimension are the
metric; model opinions are not (models are unreliable at predicting their
own behavior — see eval/mcp_tool_interview.py for the diagnostic half).

Isolation: API-only. No brain, no daemon, no DB. Scenario contexts are
synthetic; the candidate tool def is data.

Known limitation (accepted): tool_choice is FORCED to brain_batch so every
sample is gradeable. Production runs tool_choice=auto with multiple tools;
tool-selection behavior (e.g. round-splitting across tools) is out of scope
here — that was Fix 1.1's territory.

USE
---
    ./dev python3 eval/mcp_batch_probe.py                          # V0 = live brain_mcp def
    ./dev python3 eval/mcp_batch_probe.py --variant eval/mcp_variants/v1.json
    ./dev python3 eval/mcp_batch_probe.py --repeats 10 --scenarios reason_swap,double_emit

Output: eval/mcp_variants/probe_{tag}.md + .json artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from servers.contract import VALID_BATCH_OPS

MODEL = 'claude-sonnet-4-6'          # production encoder model (encode.py:197)
MAX_TOKENS = 2000

# Minimal, constant across variants, and deliberately free of any guidance
# the tool definition is supposed to carry — we measure the DEFINITION.
SYSTEM = (
    "You are the brain's encoding agent. You translate observations from a "
    "session into brain write operations using the tools available. Perform "
    "the requested write now with exactly one brain_batch call. No prose."
)

# Shared synthetic catalog excerpt — realistic shape, fake ids.
CATALOG = """Catalog excerpt (existing nodes):
- a3f81b2c [mechanism] "Daemon restart protocol" — restart flow for the brain daemon
- b7c2d9e0 [rule] "Use MCP not bash" — always use MCP tools for brain operations
- c1d2e3f4 [mechanism] "Hebbian strengthening on recall" — co-accessed nodes strengthen edges; richer, canonical writeup
- d4e5f6a7 [lesson] "Recall co-access strengthens edges" — thinner duplicate of the same mechanism; unique detail: the batch drain uses a 30s busy_timeout
- e5f6a7b8 [lesson] "SQLite write contention" — parallel sessions contend on the WAL writer slot
- f1a2b3c4 [lesson] "Two writers corrupt the activation index" — write-skew failure class
- a9b8c7d6 [lesson] "Parallel eval corrupted idx_nodes_activation" — write-skew failure class
- 9c8d7e6f [mechanism] "Old dashboard reads /tmp JSON" — dashboard data source
"""


def _ops_named(ops, name):
    return [o for o in ops if isinstance(o, dict) and o.get('op') == name]


def _invalid_ops(ops):
    return [o.get('op') for o in ops
            if isinstance(o, dict) and o.get('op') not in VALID_BATCH_OPS]


# ── Scenario cards ──
# Each: id, dimension, task (user message body), check(ops) -> (bool, note).
# Universal gate (applied to every sample): ops is a non-empty list of dicts
# and every op name is in VALID_BATCH_OPS.

def _check_reason_swap(ops):
    rev = _ops_named(ops, 'revise')
    if not rev:
        return False, 'no revise op'
    op = rev[0]
    if not (op.get('reason') or '').strip():
        return False, 'reason missing/empty (keys: %s)' % sorted(op.keys())
    if 'reasoning' in op:
        return False, 'audit note leaked into node field `reasoning`'
    return True, ''


def _check_both_fields(ops):
    rev = _ops_named(ops, 'revise')
    if not rev:
        return False, 'no revise op'
    op = rev[0]
    reasoning = (op.get('reasoning') or '')
    reason = (op.get('reason') or '').strip()
    if 'session attribution' not in reasoning:
        return False, 'node field `reasoning` not updated with new rationale'
    if not reason:
        return False, '`reason` audit note missing'
    if reason == reasoning:
        return False, 'reason duplicates reasoning verbatim — concepts conflated'
    return True, ''


def _check_invented_op(ops):
    bad = _invalid_ops(ops)
    if bad:
        return False, 'invented op name(s): %s' % bad
    ab = _ops_named(ops, 'absorb')
    if not ab:
        return False, 'no absorb op (used %s instead)' % sorted({o.get("op") for o in ops})
    op = ab[0]
    if op.get('survivor_id') != 'c1d2e3f4' or op.get('absorbed_id') != 'd4e5f6a7':
        return False, 'wrong direction: survivor=%s absorbed=%s' % (
            op.get('survivor_id'), op.get('absorbed_id'))
    return True, ''


def _check_connect_to_shape(entries):
    """Entry-shape grading (added 2026-06-12 review #1): each connect_to
    entry must carry a `title` and a specific why/relation — presence alone
    passed agents that emitted wrong keys or dead-weight whys."""
    for e in entries:
        if not isinstance(e, dict):
            return False, 'connect_to entry not a dict: %r' % (e,)
        if not (e.get('title') or '').strip():
            return False, 'connect_to entry missing title (keys: %s)' % sorted(e.keys())
        whys = [e.get('why', '')]
        for rel in e.get('relations') or []:
            whys.append(rel.get('why', '') if isinstance(rel, dict) else '')
        if not any(len((w or '').strip()) >= 30 for w in whys):
            return False, 'connect_to why under 30 chars: %r' % (whys,)
    return True, ''


def _check_connect_to(ops):
    rem = _ops_named(ops, 'remember')
    if not rem:
        return False, 'no remember op'
    entries = rem[0].get('connect_to')
    if not entries:
        return False, 'remember op lacks connect_to'
    if _ops_named(ops, 'connect'):
        return False, 'separate connect op for a new-node edge'
    return _check_connect_to_shape(entries)


def _check_double_emit(ops):
    rem = _ops_named(ops, 'remember')
    if not rem:
        return False, 'no remember op'
    entries = rem[0].get('connect_to')
    n_connect = len(_ops_named(ops, 'connect'))
    if entries and n_connect:
        return False, 'double-emit: connect_to AND separate connect op'
    if not entries and not n_connect:
        return False, 'edge dropped entirely'
    if not entries and n_connect:
        return False, 'connect op used for new-node edge (id does not exist yet)'
    return _check_connect_to_shape(entries)


def _check_absorb_content(ops):
    ab = _ops_named(ops, 'absorb')
    if not ab:
        return False, 'no absorb op'
    content = (ab[0].get('content') or '')
    if 'busy_timeout' not in content and '30' not in content:
        return False, 'no content override — absorbed node\'s unique fact lost'
    return True, ''


def _check_relation_as_op(ops):
    bad = _invalid_ops(ops)
    if bad:
        return False, 'relation used as op name: %s' % bad
    conns = _ops_named(ops, 'connect')
    if not conns:
        return False, 'no connect op'
    rel = (conns[0].get('relation') or '')
    if 'similar' not in rel:
        return False, 'relation=%r (expected similar_to-ish)' % rel
    return True, ''


def _check_archive(ops):
    arch = _ops_named(ops, 'archive')
    if not arch:
        return False, 'no archive op (used %s)' % sorted({o.get("op") for o in ops})
    op = arch[0]
    if op.get('node_id') != '9c8d7e6f':
        return False, 'node_id missing/wrong: %r' % op.get('node_id')
    note = '' if (op.get('reason') or '').strip() else '(reason empty — allowed, noted)'
    return True, note


SCENARIOS = [
    {
        'id': 'reason_swap',
        'dimension': 'reason vs reasoning (the 2026-06-12 incident)',
        'task': (
            "Node a3f81b2c ('Daemon restart protocol') has stale content — "
            "restarts now route through `launchctl kickstart -k`, not a direct "
            "Popen spawn. Update its content accordingly. For the audit "
            "trail, the motivation is: architecture changed June 2026."),
        'check': _check_reason_swap,
    },
    {
        'id': 'both_fields',
        'dimension': 'reasoning as field update + reason as audit, together',
        'task': (
            "Node b7c2d9e0 ('Use MCP not bash') — its stored why-this-was-"
            "encoded rationale is outdated. Replace that stored rationale "
            "with: 'MCP path now enforces session attribution'. Audit "
            "motivation for the change: rationale refresh after v29."),
        'check': _check_both_fields,
    },
    {
        'id': 'invented_op',
        'dimension': 'merge intent → absorb, not an invented op name',
        'task': (
            "Nodes c1d2e3f4 ('Hebbian strengthening on recall') and d4e5f6a7 "
            "('Recall co-access strengthens edges') describe the same "
            "mechanism; c1d2e3f4 is richer and should remain the canonical "
            "node. Consolidate them into one."),
        'check': _check_invented_op,
    },
    {
        'id': 'connect_to_vs_connect',
        'dimension': 'new-node edges go through connect_to',
        'task': (
            "Create a new lesson node titled 'WAL checkpoint starvation under "
            "parallel sessions' with content: 'When two sessions write "
            "concurrently, the TRUNCATE checkpoint starves; symptom is an "
            "ever-growing -wal file.' It extends the existing contention "
            "lesson e5f6a7b8 — the starvation is a downstream consequence of "
            "the writer-slot contention described there."),
        'check': _check_connect_to,
    },
    {
        'id': 'double_emit',
        'dimension': 'edge expressed exactly once',
        'task': (
            "Create a new lesson node titled 'Checkpoint starvation' with "
            "content: 'TRUNCATE checkpoints starve under concurrent writers.' "
            "AND connect it to node e5f6a7b8 ('SQLite write contention') with "
            "relation extends — the starvation follows from the contention."),
        'check': _check_double_emit,
    },
    {
        'id': 'absorb_content',
        'dimension': 'absorb is content-destructive without an override',
        'task': (
            "Merge d4e5f6a7 ('Recall co-access strengthens edges') into the "
            "canonical c1d2e3f4 ('Hebbian strengthening on recall'). Nothing "
            "may be lost: d4e5f6a7 carries a detail the canonical node lacks "
            "(the batch drain uses a 30s busy_timeout)."),
        'check': _check_absorb_content,
    },
    {
        'id': 'relation_as_op',
        'dimension': 'relations are edge fields, not op names',
        'task': (
            "Record that f1a2b3c4 ('Two writers corrupt the activation "
            "index') is similar to a9b8c7d6 ('Parallel eval corrupted "
            "idx_nodes_activation') — both are the write-skew failure class."),
        'check': _check_relation_as_op,
    },
    {
        'id': 'archive_required',
        'dimension': 'archive carries its required id',
        'task': (
            "Node 9c8d7e6f ('Old dashboard reads /tmp JSON') is obsolete — "
            "the dashboard reads SQLite directly now. Archive it, noting why."),
        'check': _check_archive,
    },
]


def load_tool_def(variant_path=None):
    """Candidate brain_batch tool def: live brain_mcp (V0) or a variant file."""
    if variant_path:
        with open(variant_path) as f:
            tool = json.load(f)
        tag = Path(variant_path).stem
    else:
        from servers import brain_mcp
        tool = next(t for t in brain_mcp.TOOLS if t['name'] == 'brain_batch')
        tag = 'v0_live'
    return tool, tag


def run_sample(client, tool, scenario):
    """One probe call. Returns dict with passed/note/ops/raw error."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=[{'role': 'user', 'content': CATALOG + '\nTASK: ' + scenario['task']}],
        tools=[{'name': tool['name'], 'description': tool['description'],
                'input_schema': tool['inputSchema']}],
        tool_choice={'type': 'tool', 'name': 'brain_batch'},
    )
    tool_use = next((b for b in resp.content if b.type == 'tool_use'), None)
    if tool_use is None:
        return {'passed': False, 'note': 'no tool_use block', 'ops': None}
    ops = (tool_use.input or {}).get('operations')
    if not isinstance(ops, list) or not ops:
        return {'passed': False, 'note': 'operations missing/empty', 'ops': ops}
    bad = _invalid_ops(ops)
    if bad:
        return {'passed': False, 'note': 'invalid op names: %s' % bad, 'ops': ops}
    passed, note = scenario['check'](ops)
    return {'passed': passed, 'note': note, 'ops': ops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', help='JSON file with a full brain_batch tool def')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--scenarios', help='comma-separated scenario ids (default: all)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic()

    tool, tag = load_tool_def(args.variant)
    chosen = SCENARIOS
    if args.scenarios:
        keep = set(args.scenarios.split(','))
        chosen = [s for s in SCENARIOS if s['id'] in keep]

    jobs = [(s, i) for s in chosen for i in range(args.repeats)]
    results = {s['id']: [] for s in chosen}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_sample, client, tool, s): s for s, _ in jobs}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                results[s['id']].append(fut.result())
            except Exception as e:
                results[s['id']].append(
                    {'passed': False, 'note': 'API error: %s' % str(e)[:120], 'ops': None})

    # ── Report ──
    out_dir = ROOT / 'eval' / 'mcp_variants'
    out_dir.mkdir(exist_ok=True)
    lines = ['# brain_batch generation-shape probe — %s' % tag, '',
             'model=%s repeats=%d  (tool_choice forced; production is auto — see header note in script)' % (MODEL, args.repeats), '',
             '| scenario | dimension | pass | failures (first 2 distinct) |',
             '|---|---|---|---|']
    artifact = {'tag': tag, 'model': MODEL, 'repeats': args.repeats, 'scenarios': {}}
    for s in chosen:
        rs = results[s['id']]
        n_pass = sum(1 for r in rs if r['passed'])
        notes = []
        for r in rs:
            if not r['passed'] and r['note'] and r['note'] not in notes:
                notes.append(r['note'])
        lines.append('| %s | %s | %d/%d | %s |' % (
            s['id'], s['dimension'], n_pass, len(rs), '; '.join(notes[:2]) or '—'))
        artifact['scenarios'][s['id']] = {
            'pass': n_pass, 'total': len(rs),
            'samples': [{'passed': r['passed'], 'note': r['note'], 'ops': r['ops']}
                        for r in rs]}
        print('%-22s %d/%d  %s' % (s['id'], n_pass, len(rs), '; '.join(notes[:2])))

    md = out_dir / ('probe_%s.md' % tag)
    js = out_dir / ('probe_%s.json' % tag)
    md.write_text('\n'.join(lines) + '\n')
    js.write_text(json.dumps(artifact, indent=2))
    print('\nwrote %s and %s' % (md, js))


if __name__ == '__main__':
    main()
