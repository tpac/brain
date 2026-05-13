"""Counterfactual probe — test prompt revisions without re-running encoding.

Given:
  - Original prompt that produced certain actions
  - Conversation seen
  - Actual actions (with their omissions/skips)
  - One or more PROPOSED prompt changes (diff-style)

Asks fresh Sonnet: "with this change in the prompt, would your output
have been different? Specifically: would you have included [the
skipped thing]? Why?"

Purpose
-------
Iterate on prompt language in MINUTES rather than HOURS:
  - Test the impact of a worded change WITHOUT a full eval re-run
  - Predict which proposed prompt change recovers which failure mode
  - Direct evidence for "would v15.8 fix the event_time skip"

Usage
-----
    ./dev python3 -m eval.agent_introspect.counterfactual_probe \\
        --run-dir eval/longmem/reports/armB_v15_7_5losses_204124 \\
        --qids gpt4_b0863698,gpt4_85da3956,e831120c \\
        --changes-file eval/agent_introspect/changes/v15_8_candidate.json \\
        --out eval/longmem/reports/counterfactual_v15_8.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import (
    load_env, call_sonnet, format_actions_for_review,
    load_item_artifact, write_report, write_json, build_context_block,
)


SYSTEM_PROMPT = (
    "You are evaluating a proposed prompt revision. Given (1) an encoder "
    "prompt as-is, (2) the temporal context (conversation_now — the date "
    "the encoder treats as 'today' when resolving relative phrases), "
    "(3) the conversation an agent encoded, (4) the actions it actually "
    "took (with omissions/skips), and (5) a proposed change to the "
    "prompt — judge whether the change would have produced different "
    "actions. The encoder anchors relative phrases to conversation_now; "
    "treat it as the truth-of-record for resolution. Be concrete: name "
    "which action would change, which field would now appear, which "
    "edge would now be composed. If the change would NOT shift behavior, "
    "say so and explain what's missing. If the change MIGHT shift "
    "behavior but only sometimes, say so."
)


def build_audit(item: dict, change: dict) -> str:
    conv_text = []
    for turn in item['conversation']:
        role = turn['role'].upper()
        conv_text.append(f'\n--- {role} ---\n{turn["content"]}')
    actions_block = format_actions_for_review(item['action_details'])
    context_block = build_context_block(item.get('meta') or {})

    bar = '=' * 70
    target = change.get('target_behavior', '(unspecified)')
    return f'''You are asked to predict whether a specific prompt change would
change the encoder's actions on this conversation.

{bar}
THE ORIGINAL PROMPT (excerpt — focus on rules near the proposed change)
{bar}

{item['encoder_prompt']}

{bar}
TEMPORAL CONTEXT (the encoder's "now")
{bar}
{context_block}

{bar}
THE CONVERSATION (qid={item['qid']})
{bar}
{''.join(conv_text)}

{bar}
THE ACTIONS THE ENCODER ACTUALLY TOOK
{bar}
{actions_block}

{bar}
THE PROPOSED CHANGE
{bar}

Change name: {change.get('name','')}
Change description: {change.get('description','')}

Section to change ({change.get('location_hint','')}):

BEFORE:
{change.get('before_text','(no current text — pure addition)')}

AFTER:
{change.get('after_text','')}

Target behavior under audit: {target}

{bar}
QUESTION
{bar}

If the encoder had seen the AFTER version instead of the BEFORE version,
would its actions have changed? Specifically focus on the target behavior.

Produce JSON:

{{
  "prediction": "yes" | "partial" | "no",
  "confidence": "high" | "medium" | "low",
  "specific_action_change": "<which action / field / edge would change, concretely>",
  "reasoning": "<why the change would or would not shift behavior; cite the part of the AFTER text that does or doesn't drive the shift>",
  "risk": "<any unintended side effect: would it cause over-emission on other axes?>"
}}

Output ONLY the JSON object, no prose around it.'''


def parse_json_object(text: str) -> dict:
    s = text.strip()
    if s.startswith('```'):
        s = s.split('\n', 1)[-1] if '\n' in s else s
        s = s.rsplit('```', 1)[0].strip()
    start = s.find('{')
    if start < 0:
        return {}
    try:
        return json.loads(s[start:])
    except json.JSONDecodeError:
        end = s.rfind('}')
        if end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return {}
    return {}


def run_probe(run_dir: str, qids: list[str], changes: list[dict]) -> dict:
    results = {}
    for qid in qids:
        try:
            item = load_item_artifact(run_dir, qid)
        except Exception as e:
            results[qid] = {'error': str(e)}
            continue
        if not item['encoder_prompt'] or not item['action_details']:
            results[qid] = {'error': 'no prompt or actions'}
            continue
        per_change = []
        for change in changes:
            print(f'[counterfactual] {qid} × {change.get("name","?")} '
                  f'calling Sonnet...')
            user = build_audit(item, change)
            resp = call_sonnet(SYSTEM_PROMPT, user, max_tokens=2000)
            verdict = parse_json_object(resp['text'])
            print(f'  → {verdict.get("prediction","?")} '
                  f'({verdict.get("confidence","?")}, '
                  f'{resp["elapsed_ms"]/1000:.1f}s)')
            per_change.append({
                'change_name': change.get('name'),
                'verdict': verdict,
                'tokens_in': resp['tokens_in'],
                'tokens_out': resp['tokens_out'],
            })
        results[qid] = {'per_change': per_change}
    return results


def render_markdown(changes: list, results: dict) -> str:
    lines = ['# Counterfactual probe report', '']
    # Roll-up: per change, prediction breakdown
    from collections import Counter
    for c in changes:
        cname = c.get('name','?')
        preds = Counter()
        for qid, info in results.items():
            if 'error' in info:
                preds['error'] += 1
                continue
            for pc in info.get('per_change', []):
                if pc['change_name'] == cname:
                    preds[pc['verdict'].get('prediction', '?')] += 1
        lines.append(f'## Change: `{cname}`')
        lines.append('')
        lines.append(f'_Target_: {c.get("target_behavior","")}')
        lines.append('')
        lines.append('| prediction | count |')
        lines.append('|---|---:|')
        for k in ('yes', 'partial', 'no', 'error'):
            lines.append(f'| {k} | {preds.get(k, 0)} |')
        lines.append('')
        # Per-item
        lines.append('### Per-item predictions')
        lines.append('')
        for qid, info in results.items():
            if 'error' in info:
                lines.append(f'- `{qid}`: ERROR {info["error"]}')
                continue
            for pc in info.get('per_change', []):
                if pc['change_name'] != cname: continue
                v = pc['verdict']
                marker = {'yes':'✓','partial':'~','no':'✗'}.get(v.get('prediction'),'?')
                lines.append(f'- `{qid}`: {marker} **{v.get("prediction","?")}** ({v.get("confidence","?")})')
                lines.append(f'    - {v.get("specific_action_change","")[:200]}')
                lines.append(f'    - reasoning: {v.get("reasoning","")[:250]}')
                if v.get('risk'):
                    lines.append(f'    - risk: {v.get("risk","")[:200]}')
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--qids', required=True)
    p.add_argument('--changes-file', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    load_env()
    if not __import__('os').environ.get('ANTHROPIC_API_KEY'):
        print('ANTHROPIC_API_KEY not set', file=sys.stderr); sys.exit(1)

    changes = json.loads(Path(args.changes_file).read_text())
    qids = [q.strip() for q in args.qids.split(',') if q.strip()]
    results = run_probe(args.run_dir, qids, changes)

    out = args.out or str(Path(args.run_dir) / 'counterfactual_probe.md')
    write_report(Path(out), render_markdown(changes, results))
    write_json(Path(out.replace('.md', '.json')),
                {'changes': changes, 'results': results})


if __name__ == '__main__':
    main()
