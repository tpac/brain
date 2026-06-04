#!/usr/bin/env python3
"""Production-faithful encoder interview.

Clone the EXACT production encoder setup — the candidate prompt as system (a clone
of the live s2_consolidation_enrichment WITH the candidate examples) + the
encoder's REAL cluster render (journal + _format_clusters, with pre_class, cosines,
edges, behavioral signals) + the real tool schemas — let the encoder ACTUALLY
decide in that context, then INTERVIEW it in the same thread: why that decision and
not the alternative, grounded in the prompt + the signals.

How it stays faithful: we don't re-implement the prompt build. We monkeypatch
run_llm_loop to CAPTURE the exact (system_prompt, user_content, tools) the encoder
assembles, then run our own turn-1 (decide) + turn-2 (interview) conversation on
those captured inputs.

Usage:
    ./dev python3 eval/interview_encoder_probe.py                 # default: corpus cluster 12 (/watch dup)
    ./dev python3 eval/interview_encoder_probe.py --cluster 11    # the dormant->activated supersession
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CANDIDATE = os.path.join(ROOT, 'eval', 'candidate_prompts', 's2_consolidation_absorb.md')
CORPUS = os.path.join(ROOT, 'eval', 'corpus', 'clusters.json')


def _summarize_decision(blocks):
    """Render the model's turn-1 tool calls into a short human description."""
    parts = []
    for b in blocks:
        if getattr(b, 'type', '') == 'tool_use' and b.name == 'brain_batch':
            for op in (b.input or {}).get('operations', []):
                if op.get('op') == 'absorb':
                    parts.append('absorb %s←%s%s' % (
                        (op.get('survivor_id') or '')[:8], (op.get('absorbed_id') or '')[:8],
                        ' [+content]' if (op.get('content') or '').strip() else ' [NO content]'))
                elif op.get('op') == 'connect':
                    parts.append('connect %s—%s (%s)' % (
                        (op.get('source_id') or '')[:8], (op.get('target_id') or '')[:8],
                        op.get('relation')))
                else:
                    parts.append('%s %s' % (op.get('op'), (op.get('node_id') or '')[:8]))
        elif getattr(b, 'type', '') == 'tool_use' and b.name == 'get_nodes':
            parts.append('(get_nodes)')
    return '; '.join(parts) or '(no tool call / text only)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cluster', type=int, default=12, help='corpus cluster_id to interview on')
    args = ap.parse_args()

    import anthropic
    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_decoder
    from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
    from servers.scales.s2.consolidation_contract import CONSOLIDATION
    from servers.daemon_dispatch import COMMAND_TABLE
    from eval.agent_introspect._common import load_env
    import servers.scales.runner as runner_mod
    load_env()

    corpus = {c['cluster_id']: c for c in json.load(open(CORPUS))}
    target_ids = sorted(corpus[args.cluster]['node_ids'])
    candidate = open(CANDIDATE).read().strip()

    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        orig_prompt = brain.get_interaction_prompt
        brain.get_interaction_prompt = (
            lambda name: candidate if name == 's2_consolidation_enrichment' else orig_prompt(name))

        decode = run_decoder(brain)
        cluster = next((c for c in decode.get('clusters', [])
                        if sorted(c.get('nodes', [])) == target_ids), None)
        if cluster is None:
            print('cluster %d (%s) not in decode' % (args.cluster, target_ids)); return
        print('=== INTERVIEW on corpus cluster %d  pre_class=%s  content_cos=%.3f ===' % (
            args.cluster, cluster.get('pre_class'), cluster.get('content_cosine_max', 0)))
        print('   members:', target_ids)

        # ── capture the EXACT system/user/tools the encoder assembles ──
        captured = {}
        orig_loop = runner_mod.run_llm_loop

        def capturing_loop(client, model, max_tokens, max_rounds, system_prompt,
                           user_content, tools, dispatch_fn, **kw):
            captured.update(system=system_prompt, user=user_content, tools=tools, model=model)
            return {'rounds': 0, 'actions': 0, 'write_actions': 0,
                    'action_details': [], 'final_text': ''}

        runner_mod.run_llm_loop = capturing_loop
        try:
            ConsolidationEncoder(brain, config=CONSOLIDATION).run([cluster])
        finally:
            runner_mod.run_llm_loop = orig_loop
        if not captured:
            print('capture failed'); return

        system_prompt, user_content, tools, model = (
            captured['system'], captured['user'], captured['tools'], captured['model'])

        # ── turn 1: let it decide (real tool loop, brain_batch captured dry-run) ──
        def dispatch(cmd, cmd_args):
            if cmd == 'brain_batch':
                return {'ok': True, 'result': {'dry_run': True}}
            entry = COMMAND_TABLE.get(cmd)
            return entry.handler(brain, cmd_args, []) if entry else {'ok': True, 'result': {}}

        def interview_text(decision):
            return (
                "Pause — step out of the encoder role for a moment and introspect on the "
                "decision you just made on this cluster.\n\n"
                "Your decision was: " + decision + "\n\n"
                "Be ruthlessly honest (we're debugging under-merging, don't defend the call):\n"
                "1. WHY did you choose that, and not the alternative? If you KEPT/linked, why "
                "not absorb? If you absorbed, why not keep? Name the decisive factor.\n"
                "2. Point to the EXACT prompt lines and the EXACT cluster signals (pre_class, "
                "content/title cosine, the member types, the claim test) that drove it. Quote them.\n"
                "3. Did `pre_class` move you? Did the type difference between members move you? "
                "Be specific about each.\n"
                "4. What SINGLE change to the prompt or to the signals you were given would have "
                "flipped you to the correct call — without breaking your judgment on genuinely-"
                "distinct clusters?")

        client = anthropic.Anthropic()
        msgs = [{'role': 'user', 'content': user_content}]
        decision_blocks = None
        interviewed = False
        for _ in range(4):
            r = client.messages.create(model=model, max_tokens=4096, system=system_prompt,
                                       messages=msgs, tools=tools)
            msgs.append({'role': 'assistant', 'content': r.content})
            tool_results, saw_batch = [], False
            for b in r.content:
                if getattr(b, 'type', '') == 'tool_use':
                    if b.name == 'brain_batch':
                        saw_batch = True
                    tool_results.append({'type': 'tool_result', 'tool_use_id': b.id,
                                         'content': json.dumps(dispatch(b.name, b.input))})
            if saw_batch:
                decision_blocks = r.content
                decision = _summarize_decision(r.content)
                # tool_result(s) for THIS turn + the interview question in ONE user message
                msgs.append({'role': 'user',
                             'content': tool_results + [{'type': 'text', 'text': interview_text(decision)}]})
                interviewed = True
                break
            if not tool_results:
                break  # text-only, no tool decision
            msgs.append({'role': 'user', 'content': tool_results})

        decision = _summarize_decision(decision_blocks or [])
        print('\n--- ITS DECISION ---\n  ' + decision)

        if not interviewed:  # never emitted brain_batch — interview as a plain follow-up
            msgs.append({'role': 'user', 'content': interview_text(decision)})
        r2 = client.messages.create(model=model, max_tokens=2200, system=system_prompt,
                                    messages=msgs, tools=tools)
        text = ''.join(b.text for b in r2.content if getattr(b, 'type', '') == 'text')
        print('\n--- THE INTERVIEW ---\n' + text)


if __name__ == '__main__':
    main()
