#!/usr/bin/env python3
"""
Benchmark: Common-Word Filter — Domain-Specific Term Detection

Tests the is_domain_specific() function against a variety of terms:
technical jargon, common English, acronyms, multi-word phrases, edge cases.

Run: python tests/bench_common_word_filter.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.text_processing import is_domain_specific, filter_domain_terms, load_common_words


# ── Test cases ───────────────────────────────────────────────────────
# (term, expected_domain_specific, description)

TEST_CASES = [
    # === Clearly domain-specific ===
    ('webhook', True, 'tech term not in common English'),
    ('daemon', True, 'Unix term'),
    ('middleware', True, 'software architecture term'),
    ('serializer', True, 'programming term'),
    ('embeddings', True, 'ML term'),
    ('tokenizer', True, 'NLP term'),
    ('webhook', True, 'web dev term'),
    ('cron', True, 'Unix scheduler'),
    ('regex', True, 'regular expression shorthand'),
    ('linter', True, 'code quality tool'),

    # === Clearly common English ===
    ('file', False, 'common word'),
    ('house', False, 'common word'),
    ('water', False, 'common word'),
    ('computer', False, 'common word'),
    ('problem', False, 'common word'),
    ('system', False, 'common word'),
    ('time', False, 'common word'),
    ('people', False, 'common word'),
    ('world', False, 'common word'),
    ('help', False, 'common word'),

    # === Acronyms — domain-specific ===
    ('DAL', True, 'data access layer'),
    ('API', True, 'application programming interface'),
    ('NLP', True, 'natural language processing'),
    ('BART', True, 'ML model'),
    ('MNLI', True, 'NLP benchmark'),
    ('GAM', True, 'Google Ad Manager'),
    ('ONNX', True, 'ML runtime'),
    ('SQL', True, 'query language'),
    ('CSS', True, 'stylesheet language'),
    ('HTML', True, 'markup language'),

    # === Acronyms — common (should NOT be domain-specific) ===
    ('OK', False, 'common abbreviation'),
    ('AM', False, 'time abbreviation'),
    ('PM', False, 'time abbreviation'),
    ('US', False, 'country abbreviation'),
    ('UK', False, 'country abbreviation'),
    ('FAQ', False, 'common abbreviation'),
    ('FYI', False, 'common abbreviation'),

    # === Multi-word — domain-specific ===
    ('recall scorer', True, 'head "scorer" not common'),
    ('supply adapter', True, 'head "adapter" not common'),
    ('brain daemon', True, '"daemon" not common'),
    ('hook chain', True, '"hook" or "chain" signals — depends on list'),
    ('precision loop', True, '"precision" likely not in 10K'),
    ('knowledge graph', True, '"graph" may be common but domain in context'),
    ('entity resolution', True, 'NLP concept'),

    # === Multi-word — common ===
    ('the file', False, 'both words common'),
    ('good idea', False, 'both words common'),
    ('new feature', False, 'both words common'),
    ('last time', False, 'both words common'),
    ('next step', False, 'both words common'),

    # === Product/proper names (capitalized) ===
    ('Clerk', True, 'product name — not in common words as proper noun'),
    ('Valinor', True, 'private entity name'),
    ('Supabase', True, 'product name'),
    ('Redis', True, 'database'),

    # === Edge cases ===
    ('', False, 'empty string'),
    ('   ', False, 'whitespace only'),
    ('a', False, 'single common letter'),
    ('x', False, 'single letter'),
]


def main():
    # Load word list stats
    common = load_common_words()
    print('Common word list: %d words loaded' % len(common))
    print()

    # Header
    print('=' * 95)
    print('Common-Word Filter Benchmark: Domain-Specific Detection')
    print('=' * 95)
    print()
    print('%-25s | %-8s | %-8s | %-6s | %s' % ('TERM', 'EXPECTED', 'ACTUAL', 'MATCH', 'DESCRIPTION'))
    print('-' * 25 + '-+-' + '-' * 8 + '-+-' + '-' * 8 + '-+-' + '-' * 6 + '-+-' + '-' * 30)

    correct = 0
    wrong = 0
    total = 0

    for term, expected, desc in TEST_CASES:
        actual = is_domain_specific(term)
        match = actual == expected
        total += 1
        if match:
            correct += 1
            status = '  ✅'
        else:
            wrong += 1
            status = '  ❌'

        exp_str = 'domain' if expected else 'common'
        act_str = 'domain' if actual else 'common'

        print('%-25s | %-8s | %-8s | %s | %s' % (
            term[:25] if term else '(empty)',
            exp_str,
            act_str,
            status,
            desc[:30]
        ))

    print()
    print('=' * 95)
    print('Accuracy: %d/%d (%.1f%%)' % (correct, total, 100 * correct / total if total else 0))
    if wrong:
        print('Mismatches: %d — review cases above marked ❌' % wrong)
    print()

    # Test filter_domain_terms
    print('filter_domain_terms() test:')
    candidates = ['the file', 'recall scorer', 'webhook', 'DAL', 'good idea', 'brain daemon', 'OK']
    filtered = filter_domain_terms(candidates)
    print('  Input:    %s' % candidates)
    print('  Filtered: %s' % filtered)
    print()

    # Performance
    print('Performance (10,000 calls on "recall scorer"):')
    t0 = time.time()
    for _ in range(10000):
        is_domain_specific('recall scorer')
    ms = (time.time() - t0) * 1000
    print('  %.1fms total (%.4fms/call)' % (ms, ms / 10000))
    print()

    # Spot-check some words in the list
    print('Spot-check — are these in the 10K list?')
    check_words = ['hook', 'chain', 'node', 'edge', 'tree', 'graph', 'brain',
                   'daemon', 'webhook', 'middleware', 'scorer', 'adapter',
                   'precision', 'recall', 'encode', 'decode', 'parse']
    for w in check_words:
        in_list = w in common
        print('  %-15s %s' % (w, '✅ common' if in_list else '❌ not in list'))


if __name__ == '__main__':
    main()
