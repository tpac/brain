"""Diff two encoder-prompt probe runs side by side, per aspect.

Use after running encoder_prompt_probe.py twice (e.g. on v14 and v15) to
see how each Sonnet's read of the prompt shifted between versions.

Usage:
    ./dev python3 eval/encoder_prompt_diff.py \
        eval/prompts/probe_s1e_v14.json \
        eval/prompts/probe_s1e_v15.json

Output: a markdown report at eval/prompts/diff_{a}_vs_{b}.md with the
two answers side-by-side per aspect, plus a plain-text "what shifted"
summary line per aspect.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('a', help='First probe JSON (e.g. v14)')
    p.add_argument('b', help='Second probe JSON (e.g. v15)')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())

    a_name = Path(a['prompt_path']).stem
    b_name = Path(b['prompt_path']).stem

    out = args.out or str(
        Path(a['prompt_path']).parent / f"diff_{a_name}_vs_{b_name}.md")

    lines = []
    lines.append(f"# Probe diff — {a_name} → {b_name}")
    lines.append('')
    lines.append(f"**A:** {a['prompt_path']} ({a['prompt_chars']:,} chars)")
    lines.append(f"**B:** {b['prompt_path']} ({b['prompt_chars']:,} chars)")
    lines.append('')

    aspect_ids = list(a['results'].keys())
    for aid in aspect_ids:
        ra = a['results'].get(aid, {})
        rb = b['results'].get(aid, {})
        title = ra.get('title') or rb.get('title') or aid
        lines.append(f"## Aspect: {title}")
        lines.append('')

        if 'error' in ra:
            lines.append(f"**A errored:** {ra['error']}")
        if 'error' in rb:
            lines.append(f"**B errored:** {rb['error']}")
        if 'error' in ra or 'error' in rb:
            lines.append('')
            lines.append('---')
            lines.append('')
            continue

        lines.append(f"### A — {a_name}")
        lines.append('')
        lines.append(ra.get('answer', '(no answer)'))
        lines.append('')

        lines.append(f"### B — {b_name}")
        lines.append('')
        lines.append(rb.get('answer', '(no answer)'))
        lines.append('')

        lines.append('---')
        lines.append('')

    Path(out).write_text('\n'.join(lines))
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
