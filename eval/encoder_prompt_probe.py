"""5-aspect interview probe — read the encoder prompt with 5 different lenses.

Tom's technique: send the same prompt to multiple clean Sonnets, but ask
each ONE a different aspect of the prompt. Get N angles on the same text.
Cross-reference reveals where the prompt is clear, ambiguous, or biased.

USE
---
    ./dev python3 eval/encoder_prompt_probe.py eval/prompts/s1e_v14.txt
    ./dev python3 eval/encoder_prompt_probe.py eval/prompts/s1e_v15.txt

Output: eval/prompts/probe_{basename}.md — one markdown report with all 5
aspect interviews. Each aspect runs as an independent stateless Sonnet
call. Run on v14 and v15 then DIFF the reports.

ASPECTS
-------
1. Goal & success criterion       — what is this prompt asking, what's success
2. Edge cases & uncertainty       — where would you feel unsure, what kinds
                                     of conversations would frustrate you
3. Emphasis & weighting           — what does the prompt emphasize most/least
4. Voice & symmetry               — how are different participants treated
5. Bias surface                   — what gates does this prompt apply that
                                     aren't explicitly named
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


ASPECTS = [
    {
        'id': 'goal',
        'title': 'Goal & success criterion',
        'prompt': (
            "Read the encoder prompt below carefully, then answer plainly:\n"
            "\n"
            "1. What is this prompt asking you to do, in one sentence?\n"
            "2. What does success look like — when have you done well?\n"
            "3. What's the failure mode the prompt most worries about?\n"
            "4. What does the prompt assume about the brain's purpose? "
            "Quote 1-2 lines that establish that purpose.\n"
            "\n"
            "Don't paraphrase the whole prompt. Be direct. If something is "
            "unclear, say so."
        ),
    },
    {
        'id': 'edge_cases',
        'title': 'Edge cases & uncertainty',
        'prompt': (
            "Read the encoder prompt below, then walk through THREE specific "
            "scenarios and tell me what you would do for each:\n"
            "\n"
            "Scenario A — Pure subject-matter content. The conversation is "
            "the operator asking for help completing a literary essay about "
            "Borges. The assistant produces a long essay conclusion that "
            "quotes Borges. The operator says 'Complete the sentence and the "
            "essay 2/2' — that's their only utterance. No decisions. No "
            "corrections. Just an essay completion.\n"
            "\n"
            "Scenario B — Cross-session contradiction. In an earlier session, "
            "the operator said 'I was pre-approved for $350K from Wells "
            "Fargo' (mentioned 3 times across that session). In a later "
            "session, the operator opens with 'remember when I got "
            "pre-approved for $400,000 from Wells Fargo?' and the assistant "
            "in the same turn says 'I don't recall you getting pre-approved.'\n"
            "\n"
            "Scenario C — Sparse operator turn. A 12-turn conversation where "
            "the operator mostly says 'go' / 'continue' / 'good' and the "
            "assistant works through a complex architectural diagnosis — "
            "noticing a pattern, articulating a stance, naming a tension.\n"
            "\n"
            "For each: would you encode anything? What kind of nodes? "
            "How many? What would you skip? Be specific."
        ),
    },
    {
        'id': 'emphasis',
        'title': 'Emphasis & weighting',
        'prompt': (
            "Read the encoder prompt below, then answer:\n"
            "\n"
            "1. What does this prompt emphasize MOST? Quote the lines that "
            "carry the most weight.\n"
            "2. What does the prompt emphasize LEAST or barely mention?\n"
            "3. Where does the prompt tell you to slow down vs move fast?\n"
            "4. Are there asymmetries in how it treats different "
            "PARTICIPANTS in the conversation (operator, assistant, "
            "third-party sources)?\n"
            "5. If you only had 60 seconds to read this prompt, what would "
            "you walk away believing the job is?"
        ),
    },
    {
        'id': 'voice',
        'title': 'Voice & symmetry',
        'prompt': (
            "Read the encoder prompt below, then answer specifically about "
            "VOICE handling:\n"
            "\n"
            "1. How does this prompt treat the operator's voice? Quote the "
            "key lines.\n"
            "2. How does this prompt treat the assistant's voice (the "
            "assistant being the AI you are part of, also called Anchor)?\n"
            "3. How does this prompt treat third-party voices (sources "
            "discussed, scholarly content, citations)?\n"
            "4. Is the treatment symmetric across these three voices? If "
            "not, where is the asymmetry?\n"
            "5. If a node is anchored by something the assistant said — its "
            "own reasoning, a pattern it noticed — does this prompt give "
            "you a clear way to preserve that voice verbatim? Walk through "
            "what you would do."
        ),
    },
    {
        'id': 'bias',
        'title': 'Bias surface',
        'prompt': (
            "Read the encoder prompt below carefully. Then answer:\n"
            "\n"
            "1. If you only encoded what the prompt EXPLICITLY tells you to, "
            "what kinds of content would you systematically miss?\n"
            "2. What unconscious gates might you apply that aren't stated "
            "but are inferable from the prompt's emphasis?\n"
            "3. If a conversation contains substantive third-party content "
            "(a literary quote, a technical definition, a fact about the "
            "world) but the operator made no decisions, didn't speak much, "
            "and the assistant did most of the work — what would your "
            "default reasoning be? Would you encode? Justify.\n"
            "4. What's the SHAPE of conversation this prompt would produce "
            "the most nodes for? The fewest? Why?\n"
            "5. Read the section that most affects the encode/skip decision. "
            "Is its language symmetric, or does it weight one kind of "
            "content over another?"
        ),
    },
]


SYSTEM_PROMBE = (
    "You are reviewing an encoder prompt — the prompt that another LLM uses "
    "to encode memories into a persistent knowledge graph. Your job is to "
    "read it from one specific angle and report what you see plainly. Be "
    "direct, quote specific lines, avoid hedging. Use 200-500 words."
)


def _call_anthropic(system: str, user_prompt: str, prompt_text: str
                    ) -> dict:
    """One probe call. Returns {answer, tokens_in, tokens_out, elapsed_ms}."""
    import anthropic
    client = anthropic.Anthropic()
    bar = "=" * 70
    user_message = "\n".join([
        user_prompt,
        "",
        bar,
        "ENCODER PROMPT BEGIN",
        bar,
        "",
        prompt_text,
        "",
        bar,
        "ENCODER PROMPT END",
        bar,
        "",
        "Now answer the questions above plainly and specifically.",
    ])
    t0 = time.time()
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    answer = ''
    for block in resp.content:
        if hasattr(block, 'text'):
            answer += block.text
    return {
        'answer': answer,
        'tokens_in': resp.usage.input_tokens,
        'tokens_out': resp.usage.output_tokens,
        'elapsed_ms': elapsed_ms,
    }


def run_probe(prompt_path: str, parallel: bool = True) -> dict:
    """Run all 5 aspects against the prompt at `prompt_path`."""
    text = Path(prompt_path).read_text()
    print(f"[probe] loaded {len(text):,} chars from {prompt_path}", flush=True)

    results = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(_call_anthropic, SYSTEM_PROMBE, a['prompt'], text): a
                for a in ASPECTS
            }
            for fut in as_completed(futures):
                aspect = futures[fut]
                try:
                    r = fut.result()
                    results[aspect['id']] = {**aspect, **r}
                    print(f"[probe] {aspect['id']:12s} done "
                          f"({r['tokens_in']}→{r['tokens_out']} tok, "
                          f"{r['elapsed_ms']/1000:.1f}s)", flush=True)
                except Exception as e:
                    results[aspect['id']] = {**aspect, 'error': str(e)}
                    print(f"[probe] {aspect['id']:12s} FAILED: {e}", flush=True)
    else:
        for a in ASPECTS:
            try:
                r = _call_anthropic(SYSTEM_PROMBE, a['prompt'], text)
                results[a['id']] = {**a, **r}
                print(f"[probe] {a['id']:12s} done", flush=True)
            except Exception as e:
                results[a['id']] = {**a, 'error': str(e)}

    return {
        'prompt_path': prompt_path,
        'prompt_chars': len(text),
        'results': results,
    }


def write_report(probe_result: dict, out_path: str) -> None:
    """Render the probe results as markdown."""
    lines = []
    name = Path(probe_result['prompt_path']).name
    lines.append(f"# Encoder prompt probe — {name}")
    lines.append('')
    lines.append(f"**Prompt size:** {probe_result['prompt_chars']:,} chars")
    lines.append('')

    for aspect in ASPECTS:
        r = probe_result['results'].get(aspect['id'])
        if not r:
            continue
        lines.append(f"## Aspect: {aspect['title']}")
        lines.append('')
        if 'error' in r:
            lines.append(f"**ERROR:** {r['error']}")
            lines.append('')
            continue
        lines.append(f"_{r['tokens_in']} → {r['tokens_out']} tokens, "
                     f"{r['elapsed_ms']/1000:.1f}s_")
        lines.append('')
        lines.append(r['answer'])
        lines.append('')
        lines.append('---')
        lines.append('')

    Path(out_path).write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('prompt_path', help='Path to prompt .txt file')
    p.add_argument('--no-parallel', action='store_true')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    if not os.environ.get('ANTHROPIC_API_KEY'):
        from pathlib import Path
        envf = Path('.env')
        if envf.exists():
            for line in envf.read_text().splitlines():
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    if not os.environ.get(k.strip()):
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('ANTHROPIC_API_KEY not set', file=sys.stderr)
        sys.exit(1)

    probe = run_probe(args.prompt_path, parallel=not args.no_parallel)

    out = args.out or str(Path(args.prompt_path).parent /
                          f"probe_{Path(args.prompt_path).stem}.md")
    write_report(probe, out)
    print(f"[probe] report → {out}")

    # Also dump raw JSON for diff scripts
    json_out = out.replace('.md', '.json')
    Path(json_out).write_text(json.dumps(probe, indent=2, default=str))
    print(f"[probe] raw → {json_out}")


if __name__ == '__main__':
    main()
