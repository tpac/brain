#!/usr/bin/env python3
"""
Benchmark: Vocabulary Extraction — Before vs After

Compares the old 4-strategy approach with the new 7-strategy + domain filter.
Run: python tests/bench_vocab_extraction.py
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from servers.text_processing import filter_domain_terms


# ── Old extraction (copy of daemon_hooks.py original) ────────────────

def extract_old(message):
    """Original 4-strategy extraction."""
    quoted = re.findall(r'["]([\w\s-]{3,30})["]', message)
    the_patterns = re.findall(
        r"\bthe\s+([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|"
        r"module|layer|loop|sequence|pipeline|system|engine|server|db|database|config|"
        r"schema|signal|node|type|map|graph|cache|queue|log|test|spec))\b",
        message, re.IGNORECASE,
    )
    action_context = re.findall(
        r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|"
        r"rewrite|add|remove|delete|move|rename|split|merge|clean)\s+(?:the\s+)?"
        r"([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)",
        message, re.IGNORECASE,
    )
    action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]
    hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", message)
    hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
        "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
        "pre-existing", "co-authored-by",
    )]

    skip_words = {
        "the", "a", "an", "this", "that", "it", "them", "is", "are",
        "was", "were", "be", "been", "do", "does", "did", "have", "has",
        "can", "could", "will", "would", "should", "may", "might",
        "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
        "thing", "stuff", "something", "everything", "nothing",
    }
    candidates = set()
    for term in quoted + the_patterns + action_context + hyphenated:
        term = term.strip().lower()
        if len(term) < 3 or len(term) > 40:
            continue
        words = term.split()
        if all(w in skip_words for w in words):
            continue
        candidates.add(term)
    return candidates


# ── New extraction (7-strategy + domain filter) ──────────────────────

def extract_new(message):
    """New 7-strategy extraction with domain filter."""
    # Strategy 1: Quoted terms
    quoted = re.findall(r'["]([\w\s-]{3,30})["]', message)

    # Strategy 2: Expanded "the/a/this X"
    the_patterns = re.findall(
        r"\b(?:the|a|an|this|that|our|my|your)\s+"
        r"([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|"
        r"module|layer|loop|sequence|pipeline|system|engine|server|db|database|"
        r"config|schema|signal|node|type|map|graph|cache|queue|log|test|spec|"
        r"worker|adapter|pattern|protocol|daemon|scorer|mixin|handler|resolver|"
        r"builder|encoder|decoder|parser|formatter|validator|serializer|"
        r"middleware|endpoint|route|trigger|listener|callback|factory|strategy|"
        r"observer|wrapper|proxy|bridge|decorator|registry|repository|mapper|"
        r"transformer|dispatcher|emitter|collector|aggregator|provider|consumer|"
        r"subscriber|publisher|context|session|token|metric|monitor|tracer|"
        r"profiler|compiler|runtime|kernel|driver|plugin|extension|toolkit|"
        r"library|framework|platform|screen|component|service|client))\b",
        message, re.IGNORECASE,
    )

    # Strategy 3: Expanded verb-object
    action_context = re.findall(
        r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|"
        r"rewrite|add|remove|delete|move|rename|split|merge|clean|implement|"
        r"configure|deploy|migrate|optimize|integrate|initialize|bootstrap|"
        r"instrument|validate|authenticate|provision|dispatch|schedule|monitor|"
        r"benchmark|evaluate|classify|extract|transform|aggregate|normalize|"
        r"cache|batch|rollback|seed|stub|mock|patch|inject|bind|resolve|"
        r"register|subscribe|emit|consume|publish)\s+(?:the\s+)?"
        r"([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)",
        message, re.IGNORECASE,
    )
    action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]

    # Strategy 4: Hyphenated
    hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", message)
    hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
        "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
        "pre-existing", "co-authored-by",
    )]

    # Strategy 5: Capitalized mid-sentence
    capitalized = re.findall(
        r'(?<=[a-z.,;:!?\s])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        message,
    )
    capitalized = [c.strip() for c in capitalized
                  if c.strip() and c.strip() not in (
                      'I', 'The', 'This', 'That', 'It', 'We', 'You',
                      'He', 'She', 'They', 'But', 'And', 'Or', 'So',
                      'If', 'When', 'What', 'How', 'Why', 'Where',
                      'Yes', 'No', 'Ok', 'Sure', 'Please', 'Thanks',
                  )]

    # Strategy 6: Backtick-wrapped
    backtick = re.findall(r'`([^`]{2,40})`', message)

    # Strategy 7: Acronyms
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', message)

    skip_words = {
        "the", "a", "an", "this", "that", "it", "them", "is", "are",
        "was", "were", "be", "been", "do", "does", "did", "have", "has",
        "can", "could", "will", "would", "should", "may", "might",
        "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
        "thing", "stuff", "something", "everything", "nothing",
        "just", "also", "very", "really", "actually", "basically",
        "like", "about", "some", "more", "here", "there", "now", "then",
    }
    raw_candidates = set()
    for term in (quoted + the_patterns + action_context + hyphenated +
                capitalized + backtick + acronyms):
        term = term.strip()
        if len(term) < 2 or len(term) > 40:
            continue
        words = term.lower().split()
        if all(w in skip_words for w in words):
            continue
        raw_candidates.add(term)

    return set(filter_domain_terms(list(raw_candidates)))


# ── Test messages ────────────────────────────────────────────────────

TEST_MESSAGES = [
    (
        'Fix the recall scorer to handle edge cases',
        'verb-object + the_patterns',
    ),
    (
        'Check if Clerk webhook fires on magic link login',
        'capitalized product name + compound terms',
    ),
    (
        'The DAL and the Brain class need refactoring',
        'acronym + capitalized class name',
    ),
    (
        'Look at `brain.recall()` and `daemon_hooks.py` for the bug',
        'backtick code references',
    ),
    (
        'We need to implement the supply adapter pattern for GAM',
        'expanded the_patterns + acronym + verb-object',
    ),
    (
        'This BART-based scorer uses MNLI labels for entailment',
        'acronyms + hyphenated compound',
    ),
    (
        'Can you check something for me?',
        'no terms expected (common message)',
    ),
    (
        'Deploy the webhook handler to production',
        'expanded verb + expanded suffix',
    ),
    (
        'Configure the Redis cache and Supabase client',
        'capitalized products + expanded suffix',
    ),
    (
        "Valinor's pitcher is a new product by a friend of mine",
        'novel private entity name',
    ),
    (
        'The precision loop needs to evaluate recall scorer output',
        'domain compounds (both words common individually)',
    ),
    (
        'Seed the brain with test data and benchmark the embeddings',
        'expanded verbs (seed, benchmark)',
    ),
    (
        'yes please do that',
        'trivial response — no terms',
    ),
    (
        'great thanks',
        'trivial response — no terms',
    ),
    (
        'I think we should use a "supply adapter" for the ad delivery abstraction layer',
        'quoted term + the_patterns',
    ),
    (
        'The hook-chain fires pre-response-recall before pre-edit-suggest',
        'hyphenated hook names',
    ),
]


def main():
    print('=' * 100)
    print('Vocabulary Extraction Benchmark: Old (4 strategies) vs New (7 strategies + domain filter)')
    print('=' * 100)
    print()

    total_old = 0
    total_new = 0
    improvements = 0

    for msg, desc in TEST_MESSAGES:
        old = extract_old(msg)
        new = extract_new(msg)

        old_str = ', '.join(sorted(old)) if old else '(none)'
        new_str = ', '.join(sorted(new)) if new else '(none)'

        added = new - old
        removed = old - new

        total_old += len(old)
        total_new += len(new)

        status = ''
        if added:
            status = '✅ +%d' % len(added)
            improvements += 1
        elif removed:
            status = '⚠️  -%d' % len(removed)
        else:
            status = '  same'

        print('MESSAGE: "%s"' % msg[:70])
        print('  [%s]' % desc)
        print('  OLD: %s' % old_str[:80])
        print('  NEW: %s' % new_str[:80])
        if added:
            print('  ADDED: %s' % ', '.join(sorted(added)))
        if removed:
            print('  REMOVED: %s' % ', '.join(sorted(removed)))
        print('  STATUS: %s' % status)
        print()

    print('=' * 100)
    print('Summary: %d messages improved out of %d' % (improvements, len(TEST_MESSAGES)))
    print('Total extractions: old=%d, new=%d (+%d)' % (total_old, total_new, total_new - total_old))
    print()

    # Performance
    test_msg = 'Check if Clerk webhook fires on magic link login with the DAL and `brain.recall()`'
    iterations = 5000

    t0 = time.time()
    for _ in range(iterations):
        extract_old(test_msg)
    old_ms = (time.time() - t0) * 1000

    t0 = time.time()
    for _ in range(iterations):
        extract_new(test_msg)
    new_ms = (time.time() - t0) * 1000

    print('Performance (%d iterations):' % iterations)
    print('  Old: %.1fms total (%.3fms/call)' % (old_ms, old_ms / iterations))
    print('  New: %.1fms total (%.3fms/call)' % (new_ms, new_ms / iterations))


if __name__ == '__main__':
    main()
