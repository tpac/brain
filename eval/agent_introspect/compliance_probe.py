"""Compliance probe — audit agent actions against named prompt rules.

Given:
  - The encoder prompt that was active
  - The conversation the agent saw
  - The actions the agent emitted (full input fields)
  - A list of RULES from the prompt to audit

Asks a fresh stateless Sonnet to JUDGE each rule against the actions, then
EXPLAIN any non-compliance — quoting either the part of the prompt that
drove its alternative choice, or the part of the conversation that made
the rule inapplicable.

Use case
--------
After a prompt-driven encoding eval surfaces failures, run compliance_probe
on the failing items. Discover whether non-compliance is:
  - intentional (rule didn't apply to this case — agent reasoned correctly)
  - unintentional (agent missed the rule)
  - contradicted (the prompt's other instructions or examples conflicted)

The third bucket is the high-leverage finding: it surfaces prompt
contradictions automatically.

Usage
-----
    ./dev python3 -m eval.agent_introspect.compliance_probe \\
        --run-dir eval/longmem/reports/armB_v15_7_5losses_204124 \\
        --qids gpt4_b0863698,gpt4_85da3956,e831120c,184da446 \\
        --rules-file eval/agent_introspect/rules/temporal_v15_7.json \\
        --out eval/longmem/reports/compliance_probe_armB.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import (
    load_env, call_sonnet, format_actions_for_review,
    load_item_artifact, write_report, write_json,
)


SYSTEM_PROMPT = (
    "You are auditing whether an encoder agent complied with specific "
    "rules from its prompt. You will be given (1) the encoder's full "
    "system prompt, (2) the conversation it encoded, (3) the actions it "
    "emitted, and (4) a list of rules to audit. For each rule, judge: "
    "DID the agent's actions comply? Answer for each rule with: status "
    "(comply|partial|skip), evidence (concrete from the actions), and "
    "reasoning. When non-compliant, quote the part of the prompt or the "
    "conversation that explains the choice — if you can find one. If "
    "the prompt CONTRADICTS itself between the rule and somewhere else, "
    "name the contradiction and quote both sides verbatim."
)


def build_audit_user_message(item: dict, rules: list[dict]) -> str:
    """Assemble the body Sonnet sees: prompt + conversation + actions + rules."""
    conv_text = []
    for turn in item['conversation']:
        role = turn['role'].upper()
        content = turn['content']
        conv_text.append(f'\n--- {role} ---\n{content}')
    actions_block = format_actions_for_review(item['action_details'])

    bar = '=' * 70
    rules_text = '\n'.join(
        f'\n  Rule {i+1} ({r["id"]}): {r["text"]}'
        for i, r in enumerate(rules))

    return f'''Question: For each of the named rules below, audit whether the
encoder's actions COMPLIED with that rule across the conversation provided.

{bar}
PART 1 — THE ENCODER'S FULL SYSTEM PROMPT
{bar}

{item['encoder_prompt']}

{bar}
PART 2 — THE CONVERSATION THE ENCODER SAW (question_id={item['qid']})
{bar}
{''.join(conv_text)}

{bar}
PART 3 — THE ACTIONS THE ENCODER EMITTED
{bar}
{actions_block}

{bar}
PART 4 — RULES TO AUDIT
{bar}
{rules_text}

Now produce, for EACH rule, a JSON object with this exact shape:

{{
  "rule_id": "<id>",
  "status": "comply" | "partial" | "skip" | "not_applicable",
  "evidence": "<concrete observation from the actions — e.g. 'node X has event_time field' or 'no node has event_time'>",
  "reasoning": "<why the encoder did or did not follow the rule>",
  "prompt_contradiction": "<if you see another part of the prompt that contradicts this rule, quote it verbatim; else null>",
  "conversation_excuse": "<if a feature of the conversation made the rule inapplicable, quote/cite; else null>"
}}

Wrap them as one JSON array, like:
[
  {{"rule_id": "...", "status": "...", ...}},
  ...
]

Output ONLY the JSON array. No prose before or after.'''


def parse_json_array(text: str) -> list:
    """Strip code fences and extract a JSON array."""
    s = text.strip()
    if s.startswith('```'):
        s = s.split('\n', 1)[-1] if '\n' in s else s
        s = s.rsplit('```', 1)[0].strip()
    # Find the first [
    start = s.find('[')
    if start < 0:
        return []
    try:
        return json.loads(s[start:])
    except json.JSONDecodeError:
        # Try to find the last ] and parse
        end = s.rfind(']')
        if end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return []
    return []


def run_probe(run_dir: str, qids: list[str], rules: list[dict]) -> dict:
    results = {}
    for qid in qids:
        print(f'[compliance] {qid} loading...')
        try:
            item = load_item_artifact(run_dir, qid)
        except Exception as e:
            print(f'  load failed: {e}')
            results[qid] = {'error': str(e)}
            continue
        if not item['encoder_prompt']:
            print(f'  no encoder prompt in interactions.jsonl — skipping')
            results[qid] = {'error': 'no encoder prompt'}
            continue
        if not item['action_details']:
            print(f'  no encoding actions — skipping')
            results[qid] = {'error': 'no actions'}
            continue

        user_msg = build_audit_user_message(item, rules)
        print(f'  [compliance] {qid} calling Sonnet '
              f'({len(user_msg)} chars context)...')
        resp = call_sonnet(SYSTEM_PROMPT, user_msg, max_tokens=4000)
        verdict = parse_json_array(resp['text'])
        print(f'  [compliance] {qid} got {len(verdict)} rule verdicts '
              f'({resp["tokens_in"]}→{resp["tokens_out"]} tok, {resp["elapsed_ms"]/1000:.1f}s)')
        results[qid] = {
            'verdicts': verdict,
            'raw_response': resp['text'],
            'tokens_in': resp['tokens_in'],
            'tokens_out': resp['tokens_out'],
            'elapsed_ms': resp['elapsed_ms'],
        }
    return results


def render_markdown(results: dict, rules: list[dict]) -> str:
    lines = ['# Compliance probe report', '']
    rule_by_id = {r['id']: r for r in rules}

    # Roll-up — compliance rate per rule
    lines.append('## Compliance rate per rule')
    lines.append('')
    lines.append('| Rule | Comply | Partial | Skip | N/A | Error |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    rule_totals = {}
    for r in rules:
        rule_totals[r['id']] = {'comply': 0, 'partial': 0, 'skip': 0,
                                 'not_applicable': 0, 'error': 0}
    for qid, data in results.items():
        if 'error' in data:
            for r in rules:
                rule_totals[r['id']]['error'] += 1
            continue
        seen_rules = set()
        for v in data.get('verdicts', []):
            rid = v.get('rule_id', '')
            if rid not in rule_totals:
                continue
            seen_rules.add(rid)
            status = v.get('status', 'error')
            if status not in rule_totals[rid]:
                status = 'error'
            rule_totals[rid][status] += 1
        # Items missing a verdict per rule → error
        for r in rules:
            if r['id'] not in seen_rules:
                rule_totals[r['id']]['error'] += 1
    for r in rules:
        t = rule_totals[r['id']]
        lines.append(f'| `{r["id"]}` | {t["comply"]} | {t["partial"]} '
                     f'| {t["skip"]} | {t["not_applicable"]} | {t["error"]} |')
    lines.append('')

    # Per-item verdicts
    for qid, data in results.items():
        lines.append(f'## `{qid}`')
        lines.append('')
        if 'error' in data:
            lines.append(f'**Error:** {data["error"]}')
            lines.append('')
            continue
        for v in data.get('verdicts', []):
            rid = v.get('rule_id', '?')
            rtext = (rule_by_id.get(rid, {}).get('text', ''))[:120]
            status = v.get('status', '?')
            mark = {'comply':'✓', 'partial':'~', 'skip':'✗', 'not_applicable':'·'}.get(status, '?')
            lines.append(f'### {mark} {rid} — {status}')
            lines.append(f'_Rule:_ {rtext}')
            lines.append('')
            lines.append(f'**Evidence:** {v.get("evidence","")}')
            lines.append('')
            lines.append(f'**Reasoning:** {v.get("reasoning","")}')
            if v.get('prompt_contradiction'):
                lines.append('')
                lines.append(f'**Prompt contradiction:** {v.get("prompt_contradiction","")}')
            if v.get('conversation_excuse'):
                lines.append('')
                lines.append(f'**Conversation excuse:** {v.get("conversation_excuse","")}')
            lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--qids', required=True,
                   help='Comma-separated question ids')
    p.add_argument('--rules-file', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    load_env()
    if not __import__('os').environ.get('ANTHROPIC_API_KEY'):
        print('ANTHROPIC_API_KEY not set', file=sys.stderr)
        sys.exit(1)

    rules = json.loads(Path(args.rules_file).read_text())
    qids = [q.strip() for q in args.qids.split(',') if q.strip()]
    results = run_probe(args.run_dir, qids, rules)

    out = args.out or str(Path(args.run_dir) / 'compliance_probe.md')
    write_report(Path(out), render_markdown(results, rules))
    write_json(Path(out.replace('.md', '.json')),
                {'rules': rules, 'results': results})


if __name__ == '__main__':
    main()
