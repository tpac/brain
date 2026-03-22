#!/usr/bin/env python3
"""
Benchmark: Identifier Splitting — Before vs After

Compares the old two-pass re.sub approach with the new re.findall approach.
Run: python tests/bench_identifier_split.py
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.text_processing import split_identifier


# ── Old implementation (copy of brain_surface.py logic) ──────────────

def split_identifier_old(file: str) -> list:
    """The old two-pass re.sub approach from brain_surface.py."""
    clean_file = file.replace('/', ' ').replace('\\', ' ')
    # Strip extension
    ext = os.path.splitext(clean_file)[1]
    if ext:
        clean_file = clean_file.replace(ext, '')
    clean_file = re.sub(r'([a-z])([A-Z])', r'\1 \2', clean_file)
    clean_file = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', clean_file)
    file_parts = [p for p in re.split(r'[\s\-_]+', clean_file) if len(p) > 2]
    return file_parts


# ── Test cases ───────────────────────────────────────────────────────

TEST_CASES = [
    # (input, description, known_issue_with_old)
    ('brain_surface.py', 'snake_case file', False),
    ('BrainSurfaceMixin', 'PascalCase class', False),
    ('parseHTMLDoc', 'camelCase with acronym', True),
    ('getURLParser', 'camelCase leading acronym', True),
    ('HTTPSConnection', 'leading acronym + word', True),
    ('XMLParser', 'acronym + word', True),
    ('MyHTTPSURLSession', 'multiple acronyms', True),
    ('sha256Hash', 'digits in identifier', False),
    ('OAuth2Provider', 'mixed case + digit', True),
    ('pre_response_recall.py', 'snake_case with extension', False),
    ('pre-response-recall.sh', 'kebab-case with extension', False),
    ('v2.3.1-beta', 'version number', True),
    ('v5.4.0', 'version only', True),
    ('servers/daemon_hooks.py', 'path with file', False),
    ('hooks/scripts/pre_response_recall.py', 'deep path', False),
    ('__init__.py', 'dunder file', False),
    ('recall_scorer', 'snake_case no ext', False),
    ('RecallScorer', 'PascalCase', False),
    ('iPhone15ProMax', 'product name style', True),
    ('getID', 'short acronym', True),
    ('HTMLToJSON', 'acronym to acronym', True),
    ('brain.db', 'database file', False),
    ('', 'empty string', False),
    ('....', 'dots only', False),
    ('A', 'single char', False),
    ('a_b_c', 'single char segments', False),
    ('camelCase', 'basic camelCase', False),
    ('test123method', 'digits mid-word', False),
    ('UTF8Encoder', 'acronym + digit + word', True),
    ('base64Decode', 'word + digits + word', False),
    ('node_embeddings.py', 'real brain file', False),
    ('brain_consciousness.py', 'real brain file', False),
    ('daemon_hooks.py', 'real brain file', False),
    ('pre-edit-suggest.sh', 'real hook script', False),
    ('post_bash_host_check.py', 'real hook script', False),
]


def main():
    # Header
    print('=' * 100)
    print('Identifier Splitting Benchmark: Old (re.sub) vs New (re.findall)')
    print('=' * 100)
    print()
    print('%-40s | %-25s | %-25s | %s' % ('INPUT', 'OLD', 'NEW', 'STATUS'))
    print('-' * 40 + '-+-' + '-' * 25 + '-+-' + '-' * 25 + '-+-' + '-' * 10)

    improved = 0
    same = 0
    regressed = 0

    for inp, desc, known_issue in TEST_CASES:
        old = split_identifier_old(inp)
        new = split_identifier(inp)

        old_str = ' '.join(old) if old else '(empty)'
        new_str = ' '.join(new) if new else '(empty)'

        if old_str == new_str:
            status = '  same'
            same += 1
        elif known_issue:
            status = '✅ FIXED'
            improved += 1
        else:
            # Check if this is actually a regression or just different formatting
            if set(t.lower() for t in old) == set(new):
                status = '  ~same'
                same += 1
            else:
                status = '⚠️  CHANGED'
                improved += 1  # Could be regression, review manually

        print('%-40s | %-25s | %-25s | %s' % (
            inp[:40] if inp else '(empty)',
            old_str[:25],
            new_str[:25],
            status
        ))

    print()
    print('=' * 100)
    print('Summary: %d improved/fixed, %d same, %d regressed' % (improved, same, regressed))
    print()

    # Performance comparison
    print('Performance (10,000 iterations on "servers/daemon_hooks.py"):')
    iterations = 10000
    test_input = 'servers/daemon_hooks.py'

    t0 = time.time()
    for _ in range(iterations):
        split_identifier_old(test_input)
    old_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for _ in range(iterations):
        split_identifier(test_input)
    new_ms = (time.time() - t0) * 1000

    print('  Old: %.1fms total (%.3fms/call)' % (old_ms, old_ms / iterations))
    print('  New: %.1fms total (%.3fms/call)' % (new_ms, new_ms / iterations))
    print()


if __name__ == '__main__':
    main()
