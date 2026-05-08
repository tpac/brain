#!/usr/bin/env python3
"""Compare classified aspects_v1.json against a hand-crafted ground truth.

Computes routing accuracy (per category, per aspect, overall) and lists
disagreements concretely so iteration on the prompt is grounded in
specifics. Strings marked UNSURE in ground truth are reported separately
— they're the cases that warrant Tom's eye, not encoder mistakes.

Usage:
    ./dev python3 scripts/eval_aspect_classifications.py
    ./dev python3 scripts/eval_aspect_classifications.py \\
        --classified eval/aspects_v1_classified.json \\
        --ground-truth eval/aspects_ground_truth.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def invert_aspects_json(data):
    """{aspect: {node_types: [...], edge_relations: [...]}} →
       {node_types: {string: [aspects]}, edge_relations: {string: [aspects]}}.

    Multi-membership: a string can appear in multiple aspect lists.
    Each value maps to a list of aspect names it's classified into.
    """
    inverted = {'node_types': {}, 'edge_relations': {}}
    for aspect_name, aspect in data.items():
        for t in aspect.get('node_types', []):
            inverted['node_types'].setdefault(t, []).append(aspect_name)
        for r in aspect.get('edge_relations', []):
            inverted['edge_relations'].setdefault(r, []).append(aspect_name)
    return inverted


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--classified', default=os.path.join(repo_root, 'eval', 'aspects_v1_classified.json'),
                        help='Encoder output to evaluate (default: eval/aspects_v1_classified.json)')
    parser.add_argument('--ground-truth', default=os.path.join(repo_root, 'eval', 'aspects_ground_truth.json'),
                        help='Hand-crafted ground truth (default: eval/aspects_ground_truth.json)')
    parser.add_argument('--show-disagreements', type=int, default=20,
                        help='Print up to N disagreements (default: 20)')
    args = parser.parse_args()

    if not os.path.exists(args.classified):
        print('ERROR: %s not found — run scripts/run_aspect_cycles_on_clone.py first.' % args.classified, file=sys.stderr)
        return 1
    if not os.path.exists(args.ground_truth):
        print('ERROR: %s not found — build the ground truth file first.' % args.ground_truth, file=sys.stderr)
        return 1

    classified = invert_aspects_json(load_json(args.classified))
    truth = load_json(args.ground_truth)
    unsure = truth.get('_unsure', {'node_types': [], 'edge_relations': []})

    # ── Compute routing accuracy ─────────────────────────────────────
    rows = []  # (category, value, encoder_aspect, truth_aspect, status)
    per_category_totals = Counter()
    per_category_correct = Counter()
    per_aspect_seen = defaultdict(int)       # truth_aspect → count
    per_aspect_correct = defaultdict(int)    # truth_aspect → count
    confusion = Counter()                     # (truth, encoder) → count
    missing_in_classified = []
    missing_in_truth = []

    for category in ('node_types', 'edge_relations'):
        truth_map = truth.get(category, {})
        classified_map = classified.get(category, {})  # value → list of aspects
        unsure_set = set(unsure.get(category, []))

        for value, truth_aspect in truth_map.items():
            if value in unsure_set:
                continue
            encoder_aspects = classified_map.get(value)
            if not encoder_aspects:
                missing_in_classified.append((category, value))
                continue
            per_category_totals[category] += 1
            per_aspect_seen[truth_aspect] += 1
            # Multi-membership: encoder is correct if the ground-truth aspect
            # appears anywhere in its set. The encoder may add secondaries —
            # those don't penalize. (Want stricter primary-match? Compare
            # encoder_aspects[0] instead.)
            if truth_aspect in encoder_aspects:
                per_category_correct[category] += 1
                per_aspect_correct[truth_aspect] += 1
                rows.append((category, value, encoder_aspects, truth_aspect, 'OK'))
            else:
                confusion[(truth_aspect, encoder_aspects[0])] += 1
                rows.append((category, value, encoder_aspects, truth_aspect, 'DISAGREE'))

        # Strings the encoder classified but ground truth didn't cover
        for value in classified_map:
            if value not in truth_map and value not in unsure_set:
                missing_in_truth.append((category, value, classified_map[value]))

    # ── Print report ────────────────────────────────────────────────
    total_correct = sum(per_category_correct.values())
    total_seen = sum(per_category_totals.values())
    accuracy = (total_correct / total_seen * 100) if total_seen else 0.0

    print('═' * 70)
    print('Aspect classification eval')
    print('  classified:   %s' % args.classified)
    print('  ground truth: %s' % args.ground_truth)
    print('═' * 70)
    print()
    print('OVERALL ROUTING ACCURACY: %d/%d = %.1f%%' % (total_correct, total_seen, accuracy))
    print()
    print('Per category:')
    for category in ('node_types', 'edge_relations'):
        total = per_category_totals[category]
        correct = per_category_correct[category]
        pct = (correct / total * 100) if total else 0
        print('  %-15s  %d/%d = %.1f%%' % (category, correct, total, pct))
    print()

    print('Per aspect (truth aspect → recall):')
    for aspect in sorted(per_aspect_seen, key=lambda a: -per_aspect_seen[a]):
        seen = per_aspect_seen[aspect]
        ok = per_aspect_correct[aspect]
        pct = (ok / seen * 100) if seen else 0
        bar = '█' * int(pct / 5)
        print('  %-25s  %3d/%3d = %5.1f%%  %s' % (aspect, ok, seen, pct, bar))
    print()

    if missing_in_classified:
        print('Strings in ground truth but NOT in classified output (%d):' % len(missing_in_classified))
        for category, value in missing_in_classified[:10]:
            print('  %s  "%s"' % (category, value))
        if len(missing_in_classified) > 10:
            print('  ... and %d more' % (len(missing_in_classified) - 10))
        print()

    if missing_in_truth:
        print('Strings classified but NOT in ground truth (%d):' % len(missing_in_truth))
        for category, value, aspect in missing_in_truth[:10]:
            print('  %s  "%s" → %s' % (category, value, aspect))
        if len(missing_in_truth) > 10:
            print('  ... and %d more' % (len(missing_in_truth) - 10))
        print()

    disagreements = [r for r in rows if r[4] == 'DISAGREE']
    if disagreements:
        print('Disagreements (encoder set vs truth primary):')
        for category, value, encoder_aspects, truth_aspect, _ in disagreements[:args.show_disagreements]:
            enc_str = '/'.join(encoder_aspects) if isinstance(encoder_aspects, list) else encoder_aspects
            print('  %s  "%s"  encoder=[%s]  truth=%s' % (category, value, enc_str, truth_aspect))
        if len(disagreements) > args.show_disagreements:
            print('  ... and %d more (raise --show-disagreements to see all)' % (
                len(disagreements) - args.show_disagreements))
        print()

    if confusion:
        print('Confusion matrix (top 10 mismatches):')
        for (truth_aspect, encoder_aspect), n in confusion.most_common(10):
            print('  truth=%-25s  encoder=%-25s  ×%d' % (truth_aspect, encoder_aspect, n))
        print()

    n_unsure = sum(len(unsure.get(c, [])) for c in ('node_types', 'edge_relations'))
    if n_unsure:
        print('UNSURE in ground truth: %d strings (excluded from accuracy)' % n_unsure)
        for category in ('node_types', 'edge_relations'):
            for value in unsure.get(category, []):
                encoder_aspects = classified.get(category, {}).get(value, [])
                enc_str = '/'.join(encoder_aspects) if encoder_aspects else '<not classified>'
                print('  %s  "%s"  encoder=[%s]' % (category, value, enc_str))
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
