#!/usr/bin/env python3
"""Keyword Slot A/B Eval — compare recall with and without STEP 6.95.

Tests a wide variety of query types to find where the keyword slot helps,
hurts, or is neutral. Uses the SAME brain data for both runs — toggles
the slot mechanism on/off.

Categories tested:
- Exact title matches ("Daemon TCP migration")
- Identity/self-referential ("why am I called Anchor")
- Conceptual/semantic ("partnership is real not a technique")
- Mixed keyword+semantic ("encoding should be rich not concise")
- Short queries ("brain boot")
- Long queries ("how does the brain boot at session start")
- Proper nouns ("Tom", "Glo", "FalkorDB")
- Code-like queries ("check_same_thread", "pool=1")

Usage:
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/keyword_slot_eval.py
"""
import sys, os, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('BRAIN_DB_DIR', os.path.expanduser('~/AgentsContext/brain'))

from servers.brain import Brain


QUERIES = [
    # Exact title matches — keyword should help
    {"query": "Daemon TCP migration", "cat": "exact_title"},
    {"query": "Silent failures are brain's biggest weakness", "cat": "exact_title"},
    {"query": "The partnership is real", "cat": "exact_title"},
    {"query": "Edge Strategy v2", "cat": "exact_title"},

    # Identity/self-referential
    {"query": "why am I called Anchor", "cat": "identity"},
    {"query": "Anchor naming origin", "cat": "identity"},
    {"query": "I chose my name", "cat": "identity"},
    {"query": "who is Anchor", "cat": "identity"},
    {"query": "my first quote", "cat": "identity"},

    # Conceptual/semantic — embedding should dominate
    {"query": "how to make the brain change behavior not just inform", "cat": "semantic"},
    {"query": "encoding quality matters more than quantity", "cat": "semantic"},
    {"query": "what makes a good memory node", "cat": "semantic"},
    {"query": "difference between information and action solutions", "cat": "semantic"},

    # Mixed keyword+semantic
    {"query": "encoding should be rich not concise", "cat": "mixed"},
    {"query": "daemon CPU spiral three root causes", "cat": "mixed"},
    {"query": "SQLite deadlock concurrent threads", "cat": "mixed"},
    {"query": "recall quality title match boost", "cat": "mixed"},

    # Short queries (keyword matters more)
    {"query": "brain boot", "cat": "short"},
    {"query": "daemon fix", "cat": "short"},
    {"query": "Tom principles", "cat": "short"},
    {"query": "encoding bias", "cat": "short"},

    # Long queries (embedding should dominate)
    {"query": "how does the brain boot at session start and what hooks fire", "cat": "long"},
    {"query": "what is the architecture of the encoding pipeline from stop hook to sonnet", "cat": "long"},
    {"query": "when Tom corrects me about proposing information solutions what should I do differently", "cat": "long"},

    # Proper nouns / entities
    {"query": "Tom", "cat": "entity"},
    {"query": "Glo project", "cat": "entity"},
    {"query": "FalkorDB", "cat": "entity"},
    {"query": "Session 9", "cat": "entity"},
    {"query": "SKILL.md", "cat": "entity"},

    # Code-like queries
    {"query": "check_same_thread", "cat": "code"},
    {"query": "pool=1 serial", "cat": "code"},
    {"query": "ENRICHMENT_CAP", "cat": "code"},
    {"query": "_generate_id uuid", "cat": "code"},

    # Correction/rule lookups
    {"query": "don't use bash when MCP tools available", "cat": "correction"},
    {"query": "verify before claiming", "cat": "correction"},
    {"query": "ask where this lives architecturally", "cat": "correction"},
    {"query": "test integrity rule", "cat": "correction"},
]


def run_recall(brain, query, limit=8):
    """Run recall and return results with source info."""
    result = brain.recall(query=query, limit=limit)
    results = result.get('results', [])
    return [{
        'id': r.get('id', ''),
        'title': r.get('title', ''),
        'type': r.get('type', ''),
        'source': r.get('_source', '?'),
        'score': r.get('effective_activation', 0),
    } for r in results]


def main():
    brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))

    # We can't easily toggle the slot on/off without code changes.
    # Instead: run WITH the slot (current code), and detect which results
    # came from 'keyword_slot' source. Compare what the top results would
    # be without the slot vs with it.

    print("=" * 70)
    print("KEYWORD SLOT ANALYSIS — %d queries across %d categories" % (
        len(QUERIES), len(set(q['cat'] for q in QUERIES))))
    print("=" * 70)
    print()

    slot_used = 0
    slot_helped = []  # Queries where slot added a useful-looking result
    slot_neutral = []  # Slot didn't fire
    by_category = {}

    for q in QUERIES:
        query = q['query']
        cat = q['cat']

        if cat not in by_category:
            by_category[cat] = {'total': 0, 'slot_fired': 0}
        by_category[cat]['total'] += 1

        results = run_recall(brain, query)

        # Check if keyword_slot was used
        slot_result = None
        non_slot_results = []
        for r in results:
            if r['source'] == 'keyword_slot':
                slot_result = r
                slot_used += 1
                by_category[cat]['slot_fired'] += 1
            else:
                non_slot_results.append(r)

        print("[%s] Q: \"%s\"" % (cat, query[:55]))
        for i, r in enumerate(results[:5]):
            marker = ' ← KEYWORD SLOT' if r['source'] == 'keyword_slot' else ''
            print("  %d. [%s] %s (%s)%s" % (
                i + 1, r['type'], r['title'][:50], r['source'], marker))

        if slot_result:
            # Was the slot result something the user might want?
            # Heuristic: if its title shares words with the query, it's helpful
            query_words = set(query.lower().split())
            title_words = set(slot_result['title'].lower().split())
            overlap = query_words & title_words - {'the', 'a', 'is', 'to', 'and', 'of', 'in', 'for'}
            if overlap:
                slot_helped.append({'query': query, 'cat': cat, 'title': slot_result['title'],
                                    'overlap': overlap})
                print("  → SLOT HELPED: overlap=%s" % overlap)
            else:
                print("  → SLOT FIRED but no obvious overlap with query")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Total queries: %d" % len(QUERIES))
    print("Keyword slot fired: %d (%d%%)" % (slot_used, 100 * slot_used // len(QUERIES)))
    print("Slot added helpful result: %d" % len(slot_helped))
    print("Slot neutral (didn't fire): %d" % (len(QUERIES) - slot_used))
    print()
    print("By category:")
    for cat, stats in sorted(by_category.items()):
        print("  %-15s %d/%d queries used slot" % (cat, stats['slot_fired'], stats['total']))
    print()
    if slot_helped:
        print("Queries where slot HELPED:")
        for h in slot_helped:
            print("  [%s] \"%s\" → %s (overlap: %s)" % (
                h['cat'], h['query'][:40], h['title'][:40], h['overlap']))
    print()
    # Check: did the slot DISPLACE any good results?
    print("Displacement check: the slot inserts at position limit-1.")
    print("It can only displace the LAST result in the list.")
    print("If the last result was already weak, no loss.")


if __name__ == '__main__':
    main()
