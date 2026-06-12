#!/usr/bin/env python3
"""Production-faithful gate for brain_batch schema changes.

eval/mcp_batch_probe.py measures generation shape in a SIMPLIFIED context
(short prompt, one tool, forced tool choice). This gate closes the gap to
production: IsolatedBrain copy -> real S2 consolidation decoder -> the
encoder's EXACT (system, user, tools) capture (the interview-probe trick) ->
a real Sonnet decide turn with tool_choice auto -> mechanical validation of
every emitted brain_batch op against contract.BATCH_OP_SPECS.

Unlike eval/interview_encoder_probe.py (which targets specific corpus
clusters for reproducible interviews), this takes the first N clusters the
decoder produces TODAY — the gate cares that production-shaped emissions are
schema-valid, not which cluster they land on. Uses the production-ACTIVE
prompt, no candidate override.

Isolation: IsolatedBrain (cleanup=True). brain_batch is dry-run — no writes
land anywhere; get_nodes executes read-only against the isolated copy.

USE
---
    ./dev python3 eval/mcp_schema_gate.py --clusters 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_ROUNDS = 4


def validate_ops(ops):
    """Mechanical check of an operations array against BATCH_OP_SPECS.
    Returns list of violation strings (empty = valid)."""
    from servers.contract import BATCH_OP_SPECS
    violations = []
    if not isinstance(ops, list) or not ops:
        return ['operations missing or empty']
    for i, op_spec in enumerate(ops):
        if not isinstance(op_spec, dict):
            violations.append('op[%d] not a dict' % i)
            continue
        op = op_spec.get('op')
        spec = BATCH_OP_SPECS.get(op)
        if spec is None:
            violations.append('op[%d] invalid op name: %r' % (i, op))
            continue
        missing = [f for f in spec['required'] if not op_spec.get(f)]
        if missing:
            violations.append('op[%d] %s missing required: %s'
                              % (i, op, ', '.join(missing)))
    return violations


def decide_turn(client, captured, brain):
    """Run the captured encoder request as a real tool loop (brain_batch
    dry-run, get_nodes live read-only). Returns (all_ops, violations, trace)."""
    messages = [{'role': 'user', 'content': captured['user']}]
    all_ops, violations, trace = [], [], []
    from servers.daemon_dispatch import COMMAND_TABLE

    for _ in range(MAX_ROUNDS):
        resp = client.messages.create(
            model=captured['model'], max_tokens=4096,
            system=captured['system'], messages=messages,
            tools=captured['tools'])
        tool_uses = [b for b in resp.content if b.type == 'tool_use']
        if not tool_uses:
            trace.append('text-only round (stop)')
            break
        messages.append({'role': 'assistant', 'content': resp.content})
        tool_results = []
        for tu in tool_uses:
            if tu.name == 'brain_batch':
                ops = (tu.input or {}).get('operations', [])
                all_ops.extend(ops)
                violations.extend(validate_ops(ops))
                trace.append('brain_batch: %s' % [o.get('op') for o in ops])
                result = {'ok': True, 'result': {'dry_run': True}}
            else:
                entry = COMMAND_TABLE.get(tu.name)
                if entry and not entry.is_write:
                    result = entry.handler(brain, dict(tu.input or {}), [])
                else:
                    result = {'ok': True, 'result': {'dry_run': True}}
                trace.append(tu.name)
            tool_results.append({'type': 'tool_result', 'tool_use_id': tu.id,
                                 'content': json.dumps(result)[:8000]})
        messages.append({'role': 'user', 'content': tool_results})
        if any(tu.name == 'brain_batch' for tu in tool_uses):
            break  # decision made — the gate doesn't need the wrap-up rounds
    return all_ops, violations, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', type=int, default=2)
    args = ap.parse_args()

    import anthropic
    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_decoder
    from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
    from servers.scales.s2.consolidation_contract import CONSOLIDATION
    from eval.agent_introspect._common import load_env
    import servers.scales.runner as runner_mod
    load_env()
    client = anthropic.Anthropic()

    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        decode = run_decoder(brain)
        clusters = decode.get('clusters', [])[:args.clusters]
        if not clusters:
            print('GATE INCONCLUSIVE: decoder produced no clusters')
            return 1

        failures = 0
        for cluster in clusters:
            captured = {}
            orig_loop = runner_mod.run_llm_loop

            def capturing_loop(client, model, max_tokens, max_rounds,
                               system_prompt, user_content, tools,
                               dispatch_fn, **kw):
                captured.update(system=system_prompt, user=user_content,
                                tools=tools, model=model)
                return {'rounds': 0, 'actions': 0, 'write_actions': 0,
                        'action_details': [], 'final_text': ''}

            runner_mod.run_llm_loop = capturing_loop
            try:
                ConsolidationEncoder(brain, config=CONSOLIDATION).run([cluster])
            finally:
                runner_mod.run_llm_loop = orig_loop
            if not captured:
                print('capture failed for cluster %s' % cluster.get('nodes'))
                failures += 1
                continue

            ops, violations, trace = decide_turn(client, captured, brain)
            label = ','.join((n or '')[:8] for n in cluster.get('nodes', []))
            print('cluster [%s]  rounds: %s' % (label, ' -> '.join(trace)))
            if violations:
                failures += 1
                for v in violations:
                    print('  VIOLATION: %s' % v)
            elif not ops:
                print('  (no brain_batch emitted — encoder chose SKIP; valid)')
            else:
                print('  %d ops, all schema-valid' % len(ops))

        print('\nGATE %s' % ('FAILED (%d cluster(s) with violations)' % failures
                             if failures else 'PASSED'))
        return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
