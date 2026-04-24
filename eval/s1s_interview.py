"""Interview a Sonnet instance loaded with the full v13 scribe prompt.

Lets us (the designers) probe how the prompt reads to a model that sees
it cold — what's clear, what's confusing, what rules seem to conflict,
what a stateless reader would carry forward.

The interviewed Sonnet gets NO tools and NO encoding context (no catalog,
no conversation window, no scout reports). We're probing the prompt's
self-explanation, not its behavior on a specific input. If we wanted to
test behavior, the smoke harness does that.

Usage:
    ./dev python3 eval/s1s_interview.py --reset
    ./dev python3 eval/s1s_interview.py "In your own words, what's your job?"
    ./dev python3 eval/s1s_interview.py "Now: what's the hardest rule to keep?"
    ./dev python3 eval/s1s_interview.py --show-history

State lives at /tmp/s1s_interview_history.jsonl between calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = Path('/tmp/s1s_interview_history.jsonl')


def _load_env():
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                k, v = k.strip(), v.strip()
                # Always set — ignore any empty-string overrides in current env
                if v:
                    os.environ[k] = v


def get_system_prompt() -> str:
    """Build the full v13 system prompt using the same path encode.py uses."""
    sys.path.insert(0, str(ROOT))
    from servers.scales.s1.encode import _build_system_prompt
    from eval.s1s_v13_prompt import extract_v13_prompt
    v13 = extract_v13_prompt()
    return _build_system_prompt(prompt_instructions=v13)


def load_history() -> list:
    if not HISTORY_PATH.exists():
        return []
    return [json.loads(l) for l in HISTORY_PATH.read_text(encoding='utf-8').splitlines()
            if l.strip()]


def save_history(messages: list) -> None:
    with HISTORY_PATH.open('w', encoding='utf-8') as f:
        for m in messages:
            f.write(json.dumps(m) + '\n')


def ask(user_message: str) -> tuple:
    """Send a message, get reply, persist. Returns (reply, usage_dict)."""
    import anthropic
    messages = load_history()
    messages.append({'role': 'user', 'content': user_message})
    system = get_system_prompt()
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=4096,
        system=system,
        messages=messages,
    )
    reply = ''.join(b.text for b in resp.content if hasattr(b, 'text'))
    messages.append({'role': 'assistant', 'content': reply})
    save_history(messages)
    usage = {
        'input_tokens': getattr(resp.usage, 'input_tokens', 0),
        'output_tokens': getattr(resp.usage, 'output_tokens', 0),
        'cache_read': getattr(resp.usage, 'cache_read_input_tokens', 0) or 0,
        'cache_creation': getattr(resp.usage, 'cache_creation_input_tokens', 0) or 0,
    }
    return reply, usage


def reset() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
    print(f'[reset] cleared {HISTORY_PATH}')


def show_history() -> None:
    msgs = load_history()
    if not msgs:
        print('(no history — use --reset to start or send a message to begin)')
        return
    print(f'=== {len(msgs)} messages in history ===\n')
    for i, m in enumerate(msgs):
        label = 'INTERVIEWER' if m['role'] == 'user' else 'SCRIBE (Sonnet)'
        print(f'── [{i+1}] {label} ──')
        print(m['content'])
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Interview a Sonnet loaded with the v13 scribe prompt.')
    parser.add_argument('message', nargs='?', help='User message to send')
    parser.add_argument('--reset', action='store_true', help='Clear history')
    parser.add_argument('--show-history', action='store_true', help='Show conversation so far')
    parser.add_argument('--system-size', action='store_true',
                        help='Print system prompt size and exit')
    args = parser.parse_args()

    _load_env()

    if args.reset:
        reset()
        return
    if args.show_history:
        show_history()
        return
    if args.system_size:
        s = get_system_prompt()
        print(f'system prompt: {len(s):,} chars')
        return
    if not args.message:
        parser.print_help()
        return

    reply, usage = ask(args.message)
    print(reply)
    print(f'\n[{usage["input_tokens"]} in / {usage["output_tokens"]} out'
          f' / {usage["cache_read"]} cached-read / {usage["cache_creation"]} cached-write]')


if __name__ == '__main__':
    main()
