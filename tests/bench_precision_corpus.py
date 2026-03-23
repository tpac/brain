"""Benchmark: Precision scorer accuracy against real-world conversation corpus.

Tests all scorer signal paths (L0 regex, L1 embeddings, L1b BART, L2 cross-signals)
across diverse use cases: engineering, philosophy, science, personal, adversarial, ambiguous.

Usage:
    python tests/bench_precision_corpus.py              # Full benchmark
    python tests/bench_precision_corpus.py --category engineering  # One category
    python tests/bench_precision_corpus.py --verbose     # Show per-turn details

Sacred system benchmark — run BEFORE and AFTER any precision pipeline change.
"""

import json
import os
import sys
import time
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.recall_scorer import (
    compute_regex_signals, compute_bart_signals,
    compute_embedding_signals, score_recall,
    evidence_to_precision, classify_followup_signal,
    determine_match_method, is_bart_ready, load_bart,
)


CORPUS_PATH = os.path.join(os.path.dirname(__file__), 'corpus', 'precision_corpus.json')


def load_corpus(category=None):
    """Load conversation corpus, optionally filtered by category."""
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    convos = data['conversations']
    if category:
        convos = [c for c in convos if c['category'] == category]
    return convos


def evaluate_turn(turn, verbose=False):
    """Run the full scoring pipeline on a single turn pair.

    Returns dict with expected vs actual signal, precision, and signal details.
    """
    followup = turn['followup']
    recalled_snippets = turn.get('recalled_snippets', {})
    recalled_titles = turn.get('recalled_titles', {})
    response = turn.get('assistant_response', '')

    # L0: Regex
    regex = compute_regex_signals(followup)

    # L1b: BART
    bart = compute_bart_signals(followup)

    # L1: Embeddings
    try:
        emb = compute_embedding_signals(recalled_snippets, recalled_titles, response, followup)
        emb_available = True
    except Exception:
        emb = {"resp_words": len(response.split()), "depth_ratio": 0, "max_resp_sim": 0,
               "max_fup_sim": 0, "sim_delta": 0, "resp_on_topic_pct": 0,
               "fup_on_topic_pct": 0, "fup_sent_count": 1}
        emb_available = False

    # Combined scoring
    evidence, confidence, reasons, n_signals = score_recall(regex, bart, emb)
    precision = evidence_to_precision(evidence, confidence)
    signal = classify_followup_signal(evidence, confidence)
    method = determine_match_method(is_bart_ready(), emb_available)

    # Compare against expected
    expected_signal = turn['expected_signal']
    signal_correct = signal == expected_signal

    precision_in_range = True
    if precision is not None:
        if 'expected_precision_min' in turn and precision < turn['expected_precision_min']:
            precision_in_range = False
        if 'expected_precision_max' in turn and precision > turn['expected_precision_max']:
            precision_in_range = False

    # Check expected signals triggered
    expected_triggers = turn.get('tests_signals', [])
    triggered = []
    missed = []
    for sig in expected_triggers:
        if sig.startswith('bart_'):
            val = bart.get(sig, 0)
            if val > 0.3:
                triggered.append(sig)
            else:
                missed.append(sig)
        elif sig.startswith('fup_'):
            if regex.get(sig, False):
                triggered.append(sig)
            else:
                missed.append(sig)
        else:
            val = regex.get(sig, 0)
            if val > 0:
                triggered.append(sig)
            else:
                missed.append(sig)

    result = {
        'signal_correct': signal_correct,
        'precision_in_range': precision_in_range,
        'expected_signal': expected_signal,
        'actual_signal': signal,
        'precision': precision,
        'evidence': round(evidence, 4),
        'confidence': round(confidence, 4),
        'method': method,
        'n_signals': n_signals,
        'top_reasons': [(r, round(w, 3)) for w, r, _ in reasons[:3]],
        'triggered_signals': triggered,
        'missed_signals': missed,
        'regex': {k: v for k, v in regex.items() if v},
        'bart_top': {k: round(v, 3) for k, v in bart.items() if isinstance(v, float) and v > 0.2},
    }

    if verbose:
        print(f"\n    Followup: {followup[:80]}...")
        print(f"    Expected: {expected_signal} | Got: {signal} {'✅' if signal_correct else '❌'}")
        print(f"    Precision: {precision} | Evidence: {evidence:.3f} | Confidence: {confidence:.3f}")
        print(f"    Method: {method} | Signals: {n_signals}")
        if reasons:
            print(f"    Top reason: {reasons[0][1]} ({reasons[0][0]:.3f})")
        if missed:
            print(f"    ⚠️  Expected signals not triggered: {missed}")

    return result


def run_benchmark(category=None, verbose=False):
    """Run full benchmark against corpus."""
    convos = load_corpus(category)

    print("=" * 70)
    print("Precision Scorer Benchmark — Real-World Conversation Corpus")
    print("=" * 70)
    print(f"BART available: {is_bart_ready()}")
    print(f"Conversations: {len(convos)}")

    total_turns = sum(len(c['turns']) for c in convos)
    print(f"Total turn pairs: {total_turns}")
    print()

    # Aggregate results
    all_results = []
    by_category = {}
    signal_confusion = {'positive': {}, 'negative': {}, 'neutral': {}, 'uncertain': {}}

    for convo in convos:
        cat = convo['category']
        if cat not in by_category:
            by_category[cat] = {'correct': 0, 'total': 0, 'in_range': 0, 'precisions': []}

        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  [{cat}] {convo['id']}: {convo['description']}")

        for turn in convo['turns']:
            t0 = time.perf_counter()
            result = evaluate_turn(turn, verbose=verbose)
            result['latency_ms'] = (time.perf_counter() - t0) * 1000
            result['conversation_id'] = convo['id']
            result['category'] = cat

            all_results.append(result)
            by_category[cat]['total'] += 1
            if result['signal_correct']:
                by_category[cat]['correct'] += 1
            if result['precision_in_range']:
                by_category[cat]['in_range'] += 1
            if result['precision'] is not None:
                by_category[cat]['precisions'].append(result['precision'])

            # Confusion matrix
            expected = result['expected_signal']
            actual = result['actual_signal']
            signal_confusion[expected][actual] = signal_confusion[expected].get(actual, 0) + 1

    # ── Summary ──
    total_correct = sum(r['signal_correct'] for r in all_results)
    total_in_range = sum(r['precision_in_range'] for r in all_results)
    latencies = [r['latency_ms'] for r in all_results]

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\nOverall Signal Accuracy: {total_correct}/{total_turns} ({total_correct*100//total_turns}%)")
    print(f"Overall Precision In Range: {total_in_range}/{total_turns} ({total_in_range*100//total_turns}%)")
    print(f"Latency: avg {statistics.mean(latencies):.1f}ms, p95 {sorted(latencies)[int(len(latencies)*0.95)]:.1f}ms")

    print(f"\nBy Category:")
    for cat, stats in sorted(by_category.items()):
        acc = stats['correct'] * 100 // stats['total'] if stats['total'] else 0
        rng = stats['in_range'] * 100 // stats['total'] if stats['total'] else 0
        avg_p = statistics.mean(stats['precisions']) if stats['precisions'] else 0
        print(f"  {cat:15s}  signal={acc:3d}%  range={rng:3d}%  avg_precision={avg_p:.2f}  n={stats['total']}")

    print(f"\nConfusion Matrix (expected → actual):")
    signals = ['positive', 'negative', 'neutral', 'uncertain']
    header = f"  {'expected':12s} " + " ".join(f"{s:>10s}" for s in signals)
    print(header)
    for expected in signals:
        row = signal_confusion.get(expected, {})
        if any(row.values()):
            counts = " ".join(f"{row.get(s, 0):10d}" for s in signals)
            print(f"  {expected:12s} {counts}")

    # Missed signal analysis
    all_missed = []
    for r in all_results:
        all_missed.extend(r.get('missed_signals', []))
    if all_missed:
        from collections import Counter
        missed_counts = Counter(all_missed).most_common(10)
        print(f"\nMost Frequently Missed Expected Signals:")
        for sig, count in missed_counts:
            print(f"  {sig}: {count}x")

    # Failures
    failures = [r for r in all_results if not r['signal_correct']]
    if failures:
        print(f"\nSignal Mismatches ({len(failures)}):")
        for f in failures:
            print(f"  {f['conversation_id']}: expected={f['expected_signal']} got={f['actual_signal']} "
                  f"(evidence={f['evidence']}, confidence={f['confidence']})")

    print(f"\n{'=' * 70}")
    overall_pass = total_correct * 100 // total_turns >= 75
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'} (target: ≥75% signal accuracy)")
    print(f"{'=' * 70}")

    return {
        'total_turns': total_turns,
        'signal_accuracy': total_correct / total_turns if total_turns else 0,
        'precision_in_range': total_in_range / total_turns if total_turns else 0,
        'by_category': by_category,
        'failures': failures,
        'passed': overall_pass,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Precision scorer benchmark')
    parser.add_argument('--category', type=str, help='Filter by category')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-turn details')
    args = parser.parse_args()

    # Ensure BART is loaded for full evaluation
    print("Loading BART model...")
    load_bart()
    print(f"BART ready: {is_bart_ready()}")

    results = run_benchmark(category=args.category, verbose=args.verbose)
    return 0 if results['passed'] else 1


if __name__ == '__main__':
    exit(main())
