#!/usr/bin/env python3
"""
Benchmark: Sentence Splitting — Before (re.split) vs After (pySBD)

Run: python tests/bench_sentence_split.py
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.text_processing import split_sentences


def split_old(text):
    """Original naive re.split approach."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


TEST_CASES = [
    # (input, description, expected_sentence_count)
    (
        'I used brain.recall() to find it. It worked great.',
        'code reference with dot',
        2,
    ),
    (
        'Check self.conn.execute() for the query. Also look at os.path.join.',
        'multiple dotted code refs',
        2,
    ),
    (
        "We're on v2.3.1 now. The update fixed it.",
        'version number with dots',
        2,
    ),
    (
        'Mr. Smith said it was fine. Dr. Jones agreed.',
        'abbreviations with periods',
        2,
    ),
    (
        'See https://example.com/path.html for details. It has everything.',
        'URL with dots',
        2,
    ),
    (
        'Use e.g. this case and i.e. that case. Both work.',
        'e.g. and i.e. abbreviations',
        2,
    ),
    (
        'What?! Really?! That is amazing!',
        'multiple exclamation/question marks',
        3,
    ),
    (
        '',
        'empty string',
        0,
    ),
    (
        'No punctuation here',
        'no punctuation',
        1,
    ),
    (
        'Fix bug in daemon_hooks.py. The vocab detection fails.',
        'file extension .py followed by sentence',
        2,
    ),
    (
        'The score was 0.85 on precision. That is good enough.',
        'decimal number',
        2,
    ),
    (
        'thinking... maybe we should try it.',
        'ellipsis',
        1,
    ),
    (
        'brain.recall() found 3 results. brain.suggest() found 5.',
        'two code refs as sentence starts',
        2,
    ),
    (
        'A single sentence with no period',
        'no ending punctuation',
        1,
    ),
    (
        'First. Second. Third.',
        'three short sentences',
        3,
    ),
    (
        'The recall_scorer.py file in servers/ handles precision. It uses BART.',
        'file path + acronym',
        2,
    ),
]


def main():
    print('=' * 100)
    print('Sentence Splitting Benchmark: Old (re.split) vs New (pySBD + protection)')
    print('=' * 100)
    print()

    correct_old = 0
    correct_new = 0

    for text, desc, expected_count in TEST_CASES:
        old = split_old(text)
        new = split_sentences(text)

        old_ok = len(old) == expected_count
        new_ok = len(new) == expected_count

        if old_ok:
            correct_old += 1
        if new_ok:
            correct_new += 1

        status = ''
        if new_ok and not old_ok:
            status = '✅ FIXED'
        elif new_ok and old_ok:
            status = '  both ok'
        elif not new_ok and old_ok:
            status = '⚠️  REGRESSED'
        else:
            status = '  both wrong'

        print('INPUT: "%s"' % text[:70])
        print('  [%s] Expected: %d sentences' % (desc, expected_count))
        print('  OLD (%d): %s' % (len(old), old[:3]))
        print('  NEW (%d): %s' % (len(new), new[:3]))
        print('  %s' % status)
        print()

    total = len(TEST_CASES)
    print('=' * 100)
    print('Accuracy: Old=%d/%d (%.0f%%), New=%d/%d (%.0f%%)' % (
        correct_old, total, 100 * correct_old / total,
        correct_new, total, 100 * correct_new / total,
    ))
    print()

    # Performance
    test_text = 'I used brain.recall() to find v2.3.1 info. Mr. Smith confirmed. See https://example.com.'
    iterations = 5000

    t0 = time.time()
    for _ in range(iterations):
        split_old(test_text)
    old_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for _ in range(iterations):
        split_sentences(test_text)
    new_ms = (time.time() - t0) * 1000

    print('Performance (%d iterations):' % iterations)
    print('  Old: %.1fms total (%.3fms/call)' % (old_ms, old_ms / iterations))
    print('  New: %.1fms total (%.3fms/call)' % (new_ms, new_ms / iterations))


if __name__ == '__main__':
    main()
