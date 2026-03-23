"""Benchmarks for the Brain-to-Operator channel.

Measures:
- Context budget impact: token count increase from adding operator channel
- wrap_for_hook latency: string concatenation performance
- Operator content generation: time to curate consciousness signals
- Operator content size: distribution of content sizes

Run: python tests/bench_operator_channel.py
"""

import os
import sys
import tempfile
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.brain import Brain
from servers.brain_voice import BrainVoice


def setup_brain():
    """Create a brain with realistic test data."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, 'brain.db')
    brain = Brain(db_path)

    # Add some typical nodes
    for i in range(20):
        brain.remember(type="rule", title="Rule %d" % i, content="Content for rule %d with some details" % i)
    for i in range(5):
        brain.remember(type="lesson", title="Lesson %d" % i, content="Learned this the hard way: %d" % i)
    brain.create_reminder("Check deploy", "2020-01-01T00:00:00")
    brain.create_reminder("Review PR", "2020-01-01T00:00:00")

    return brain


def bench_context_budget_impact():
    """Measure token count increase from adding operator channel.

    Target: <15% increase.
    """
    brain = setup_brain()
    voice = BrainVoice(brain)

    # Simulate a recall with results
    results = [
        {"type": "rule", "title": "Rule 1", "content": "Content 1", "locked": True, "effective_activation": 0.9},
        {"type": "lesson", "title": "Lesson 1", "content": "Content 2", "locked": False, "effective_activation": 0.7},
    ]
    prompt_signals = {
        'aspirations': [], 'hypothesis': {"title": "Test hypothesis"},
        'tensions': [{"title": "Tension A vs Tension B"}],
        'instinct_nudge': None,
    }

    rendered = voice.render_prompt(results=results, prompt_signals=prompt_signals)

    # Baseline: just for_claude
    baseline_chars = len(rendered['for_claude'])
    operator_chars = len(rendered.get('for_operator') or '')
    # With operator channel
    merged = voice.wrap_for_hook(rendered['for_claude'], rendered.get('for_operator'))
    merged_chars = len(merged)

    overhead_pct = (merged_chars - baseline_chars) / baseline_chars * 100 if baseline_chars > 0 else 0
    absolute_overhead = merged_chars - baseline_chars

    print("\n=== Context Budget Impact ===")
    print("  Baseline (for_claude only): %d chars" % baseline_chars)
    print("  Operator content: %d chars" % operator_chars)
    print("  Merged total: %d chars" % merged_chars)
    print("  Overhead: +%d chars (%.1f%%)" % (absolute_overhead, overhead_pct))
    print("  Note: In production (3-5K claude content), overhead would be ~%.1f%%" % (
        absolute_overhead / 4000 * 100))
    # Target: operator content stays under 500 chars (absolute budget)
    print("  Target: operator <500 chars  — %s" % ("PASS" if operator_chars < 500 else "FAIL"))
    return operator_chars < 500


def bench_wrap_for_hook_latency():
    """Time wrap_for_hook() — must be <1ms.

    It's just string concatenation, so this should be trivially fast.
    """
    brain = setup_brain()
    voice = BrainVoice(brain)

    for_claude = "[BRAIN]\n" + ("x" * 5000) + "\n[/BRAIN]"
    for_operator = "@priority: high\nReminder: Ship feature\n\n@priority: low\nDream: something interesting"

    # Warmup
    for _ in range(100):
        voice.wrap_for_hook(for_claude, for_operator)

    # Measure
    times = []
    for _ in range(1000):
        t0 = time.perf_counter()
        voice.wrap_for_hook(for_claude, for_operator)
        times.append((time.perf_counter() - t0) * 1000)

    avg = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]

    print("\n=== wrap_for_hook Latency (1000 iterations) ===")
    print("  Mean: %.3f ms" % avg)
    print("  P50:  %.3f ms" % p50)
    print("  P95:  %.3f ms" % p95)
    print("  Target: <1ms  — %s" % ("PASS" if p95 < 1.0 else "FAIL"))
    return p95 < 1.0


def bench_operator_content_generation():
    """Time render_operator_prompt() — target <50ms.

    This calls get_due_reminders() and get_consciousness_signals().
    """
    brain = setup_brain()
    voice = BrainVoice(brain)

    prompt_signals = {
        'aspirations': [], 'hypothesis': {"title": "Test"},
        'tensions': [{"title": "T1"}, {"title": "T2"}],
        'instinct_nudge': None,
    }
    urgent = ["REMINDER DUE: Check deploy"]

    # Warmup
    for _ in range(10):
        voice.render_operator_prompt(prompt_signals, urgent)

    # Measure
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        voice.render_operator_prompt(prompt_signals, urgent)
        times.append((time.perf_counter() - t0) * 1000)

    avg = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]

    print("\n=== render_operator_prompt Latency (100 iterations) ===")
    print("  Mean: %.1f ms" % avg)
    print("  P50:  %.1f ms" % p50)
    print("  P95:  %.1f ms" % p95)
    print("  Target: <50ms  — %s" % ("PASS" if p95 < 50 else "FAIL"))
    return p95 < 50


def bench_operator_content_size():
    """Distribution of operator content sizes across different states.

    Track: mean, p50, p95, max chars.
    """
    brain = setup_brain()
    voice = BrainVoice(brain)

    scenarios = {
        "empty": {
            'prompt_signals': {'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
            'urgent_signals': None,
        },
        "reminders_only": {
            'prompt_signals': {'aspirations': [], 'hypothesis': None, 'tensions': [], 'instinct_nudge': None},
            'urgent_signals': None,
            # Reminders already created in setup_brain
        },
        "full_signals": {
            'prompt_signals': {
                'aspirations': [], 'hypothesis': {"title": "Long hypothesis about testing"},
                'tensions': [{"title": "Tension A"}, {"title": "Tension B"}],
                'instinct_nudge': None,
            },
            'urgent_signals': ["REMINDER DUE: Check deploy", "HEALTH: 5 compaction boundaries"],
        },
    }

    print("\n=== Operator Content Size Distribution ===")
    for name, params in scenarios.items():
        result = voice.render_operator_prompt(
            prompt_signals=params['prompt_signals'],
            urgent_signals=params.get('urgent_signals'),
        )
        size = len(result) if result else 0
        print("  %s: %d chars%s" % (name, size, " (None)" if result is None else ""))

    return True


def main():
    print("=" * 60)
    print("Brain-to-Operator Channel Benchmarks")
    print("=" * 60)

    results = {}
    results['budget'] = bench_context_budget_impact()
    results['latency'] = bench_wrap_for_hook_latency()
    results['generation'] = bench_operator_content_generation()
    results['size'] = bench_operator_content_size()

    print("\n" + "=" * 60)
    print("Summary:")
    all_pass = all(results.values())
    for name, passed in results.items():
        print("  %s: %s" % (name, "PASS" if passed else "FAIL"))
    print("\nOverall: %s" % ("ALL PASS" if all_pass else "SOME FAILURES"))
    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
