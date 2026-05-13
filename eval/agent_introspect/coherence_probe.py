"""Coherence probe — audit a prompt for internal contradictions.

Given a prompt, asks a fresh Sonnet to find:
  - Direct contradictions (rule A says X, section B says NOT X)
  - Soft conflicts (rule says X, example violates X, both implicit)
  - Ambiguities (rule could be read two valid ways)
  - Priority gaps (two rules both apply, no guidance on which wins)
  - Stale examples (worked examples that don't demonstrate stated rules)

This is the test of "the rules are right but examples contradict them" —
the hypothesis the compliance probe couldn't falsify on its own.

Usage
-----
    ./dev python3 -m eval.agent_introspect.coherence_probe \\
        --prompt eval/prompts/s1e_v15_7.txt \\
        --out eval/longmem/reports/coherence_v15_7.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.agent_introspect._common import (
    load_env, call_sonnet, write_report, write_json,
)


SYSTEM_PROMPT = (
    "You are auditing a prompt for internal coherence. The prompt is one "
    "long system prompt for an encoder agent. Find every place where the "
    "prompt contradicts itself, conflicts in priority, leaves a rule "
    "ambiguous, or shows worked examples that don't demonstrate (or "
    "violate) the stated rules. Be exhaustive but precise — every finding "
    "must quote both sides verbatim. Order findings by severity (high "
    "= rule directly contradicted by example or another rule)."
)


def build_audit(prompt_text: str) -> str:
    bar = '=' * 70
    return f'''Audit the prompt below for internal coherence. Specifically find:

1. **Direct contradictions** — rule A says X, section B says NOT X.
2. **Stale examples** — worked examples that violate or don't demonstrate
   stated rules (specifically: rule says "every node of type X must
   have field Y", example shows a type X node WITHOUT field Y).
3. **Priority gaps** — two rules both applicable, no statement of
   which wins.
4. **Ambiguities** — rule readable two valid ways.

For each finding produce JSON with this shape:

{{
  "severity": "high" | "medium" | "low",
  "kind": "contradiction" | "stale_example" | "priority_gap" | "ambiguity",
  "rule_quote": "<verbatim rule text, with line if discernible>",
  "rule_location_hint": "<section name or first words to help find it>",
  "conflicting_quote": "<verbatim conflicting text or example>",
  "conflicting_location_hint": "<section/example name>",
  "explanation": "<one-sentence what an agent would do because of this>"
}}

Output ONE JSON array containing all findings, sorted by severity.
Output ONLY the array, no prose.

{bar}
PROMPT TO AUDIT
{bar}

{prompt_text}

{bar}
END PROMPT
{bar}

Return the JSON array now.'''


def parse_json_array(text: str) -> list:
    s = text.strip()
    if s.startswith('```'):
        s = s.split('\n', 1)[-1] if '\n' in s else s
        s = s.rsplit('```', 1)[0].strip()
    start = s.find('[')
    if start < 0:
        return []
    try:
        return json.loads(s[start:])
    except json.JSONDecodeError:
        end = s.rfind(']')
        if end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return []
    return []


def render_markdown(prompt_path: str, findings: list, raw: str) -> str:
    lines = [f'# Coherence audit — {prompt_path}', '']
    lines.append(f'**Findings:** {len(findings)}')
    lines.append('')
    from collections import Counter
    by_sev = Counter(f.get('severity','?') for f in findings)
    by_kind = Counter(f.get('kind','?') for f in findings)
    lines.append('| severity | count |')
    lines.append('|---|---:|')
    for s in ('high', 'medium', 'low'):
        lines.append(f'| {s} | {by_sev.get(s, 0)} |')
    lines.append('')
    lines.append('| kind | count |')
    lines.append('|---|---:|')
    for k, n in by_kind.most_common():
        lines.append(f'| {k} | {n} |')
    lines.append('')

    sev_order = {'high': 0, 'medium': 1, 'low': 2}
    sorted_f = sorted(findings, key=lambda f: sev_order.get(f.get('severity'), 9))
    for i, f in enumerate(sorted_f, start=1):
        lines.append(f'## {i}. [{f.get("severity","?").upper()}] {f.get("kind","?")}')
        lines.append('')
        lines.append(f'**Rule** ({f.get("rule_location_hint","")}):')
        lines.append(f'> {f.get("rule_quote","")}')
        lines.append('')
        lines.append(f'**Conflicting** ({f.get("conflicting_location_hint","")}):')
        lines.append(f'> {f.get("conflicting_quote","")}')
        lines.append('')
        lines.append(f'**Effect:** {f.get("explanation","")}')
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prompt', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    load_env()
    if not __import__('os').environ.get('ANTHROPIC_API_KEY'):
        print('ANTHROPIC_API_KEY not set', file=sys.stderr); sys.exit(1)

    prompt_text = Path(args.prompt).read_text()
    print(f'[coherence] auditing {args.prompt} ({len(prompt_text)} chars)')
    user = build_audit(prompt_text)
    resp = call_sonnet(SYSTEM_PROMPT, user, max_tokens=6000)
    findings = parse_json_array(resp['text'])
    print(f'[coherence] {len(findings)} findings '
          f'({resp["tokens_in"]}→{resp["tokens_out"]} tok, {resp["elapsed_ms"]/1000:.1f}s)')

    out = args.out or args.prompt.replace('.txt', '_coherence.md')
    write_report(Path(out), render_markdown(args.prompt, findings, resp['text']))
    write_json(Path(out.replace('.md', '.json')),
                {'prompt': args.prompt, 'findings': findings,
                 'tokens_in': resp['tokens_in'], 'tokens_out': resp['tokens_out']})


if __name__ == '__main__':
    main()
