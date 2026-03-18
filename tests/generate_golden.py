#!/usr/bin/env python3
"""
tmemory — Golden Dataset Generator

Mines a real brain.db to auto-generate query/expected-recall test cases.

Strategy:
  1. Find high-signal nodes (locked, high access, strong edges)
  2. Generate natural-language queries that SHOULD retrieve those nodes
  3. Map each query to expected relevant node IDs with graded relevance
  4. Include edge cases: intent detection, temporal, type-specific queries

Output: golden_dataset.json with test cases for the eval harness.
"""

import sqlite3
import json
import os
import re
import sys
from typing import List, Dict, Any, Optional


def generate_golden_dataset(db_path: str, output_path: str = None) -> List[Dict[str, Any]]:
    """
    Mine brain.db and generate golden query/expected-recall test cases.

    Returns list of test cases, each with:
      - id: unique test case ID
      - query: natural language query
      - category: test category (keyword, intent, type_filter, hub, edge_case)
      - expected_relevant: dict of node_id → relevance_grade (2=highly, 1=relevant)
      - min_hit_rate_at_10: minimum acceptable hit rate (some queries are harder)
      - description: what this test validates
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    test_cases = []

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 1: Keyword/Title Match Tests
    # High-access locked nodes should be findable by their title keywords
    # ═══════════════════════════════════════════════════════════

    locked_nodes = conn.execute('''
        SELECT id, type, title, keywords, access_count
        FROM nodes WHERE locked = 1
        ORDER BY access_count DESC LIMIT 25
    ''').fetchall()

    for node in locked_nodes:
        title = node['title'] or ''
        node_id = node['id']
        node_type = node['type']

        # Extract a meaningful query phrase from the title
        query = _extract_query_from_title(title, node_type)
        if not query or len(query) < 5:
            continue

        # Find related nodes via edges (they should also appear in results)
        related = conn.execute('''
            SELECT target_id FROM edges WHERE source_id = ? AND weight >= 0.5
            UNION
            SELECT source_id FROM edges WHERE target_id = ? AND weight >= 0.5
        ''', (node_id, node_id)).fetchall()

        relevance = {node_id: 2}  # The primary node is highly relevant
        for r in related[:5]:  # Cap related to avoid noise
            relevance[r[0]] = 1

        test_cases.append({
            'id': f'kw_{node_id[:8]}',
            'query': query,
            'category': 'keyword',
            'expected_relevant': relevance,
            'min_hit_rate_at_10': 1.0,  # Primary node must be in top 10
            'description': f'Keyword recall for [{node_type}]: {title[:60]}',
        })

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 2: Intent Detection Tests
    # Queries with specific intent patterns should boost the right node types
    # ═══════════════════════════════════════════════════════════

    # Decision lookup intent
    decisions = conn.execute('''
        SELECT id, title, keywords FROM nodes
        WHERE type = 'decision' AND locked = 1
        ORDER BY access_count DESC LIMIT 5
    ''').fetchall()

    for dec in decisions:
        title = dec['title'] or ''
        topic = _extract_topic(title)
        if not topic:
            continue

        test_cases.append({
            'id': f'intent_dec_{dec["id"][:8]}',
            'query': f'what did we decide about {topic}',
            'category': 'intent_decision',
            'expected_relevant': {dec['id']: 2},
            'min_hit_rate_at_10': 1.0,
            'description': f'Decision lookup intent: {title[:60]}',
        })

    # Rule/how-to intent
    rules = conn.execute('''
        SELECT id, title FROM nodes
        WHERE type = 'rule' AND locked = 1
        ORDER BY access_count DESC LIMIT 5
    ''').fetchall()

    for rule in rules:
        title = rule['title'] or ''
        topic = _extract_topic(title)
        if not topic:
            continue

        test_cases.append({
            'id': f'intent_howto_{rule["id"][:8]}',
            'query': f'how should we handle {topic}',
            'category': 'intent_howto',
            'expected_relevant': {rule['id']: 2},
            'min_hit_rate_at_10': 0.8,
            'description': f'How-to intent for rule: {title[:60]}',
        })

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 3: Type-Filtered Recall Tests
    # Querying with type filter should only return matching types
    # ═══════════════════════════════════════════════════════════

    for node_type in ['rule', 'decision', 'procedure']:
        typed_nodes = conn.execute('''
            SELECT id, title FROM nodes
            WHERE type = ? AND locked = 1
            ORDER BY access_count DESC LIMIT 3
        ''', (node_type,)).fetchall()

        if not typed_nodes:
            continue

        # Use keywords from these nodes as query
        kw_parts = []
        relevance = {}
        for n in typed_nodes:
            title = n['title'] or ''
            words = _extract_topic(title)
            if words:
                kw_parts.append(words)
            relevance[n['id']] = 2

        if kw_parts:
            test_cases.append({
                'id': f'type_{node_type}',
                'query': kw_parts[0],
                'category': 'type_filter',
                'type_filter': [node_type],
                'expected_relevant': relevance,
                'min_hit_rate_at_10': 0.5,
                'description': f'Type-filtered recall for {node_type} nodes',
            })

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 4: Hub Node Tests
    # High-edge-count nodes (hubs) should be reachable via their connections
    # ═══════════════════════════════════════════════════════════

    hubs = conn.execute('''
        SELECT n.id, n.type, n.title, COUNT(*) as edge_count
        FROM nodes n
        JOIN edges e ON (n.id = e.source_id OR n.id = e.target_id)
        WHERE n.locked = 1
        GROUP BY n.id
        ORDER BY edge_count DESC
        LIMIT 5
    ''').fetchall()

    for hub in hubs:
        title = hub['title'] or ''
        topic = _extract_topic(title)
        if not topic:
            continue

        # Hub should appear in results, plus some of its neighbors
        neighbors = conn.execute('''
            SELECT target_id FROM edges WHERE source_id = ?
            UNION
            SELECT source_id FROM edges WHERE target_id = ?
            LIMIT 10
        ''', (hub['id'], hub['id'])).fetchall()

        relevance = {hub['id']: 2}
        for n in neighbors[:5]:
            relevance[n[0]] = 1

        test_cases.append({
            'id': f'hub_{hub["id"][:8]}',
            'query': topic,
            'category': 'hub',
            'expected_relevant': relevance,
            'min_hit_rate_at_10': 1.0,
            'description': f'Hub recall: [{hub["type"]}] {title[:60]} ({hub["edge_count"]} edges)',
        })

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 5: Edge Cases / Stress Tests
    # ═══════════════════════════════════════════════════════════

    # 5a: Empty/garbage query — should not crash, should return something
    test_cases.append({
        'id': 'edge_empty',
        'query': '',
        'category': 'edge_case',
        'expected_relevant': {},  # No specific expectations
        'min_hit_rate_at_10': 0.0,
        'description': 'Empty query should not crash',
        'expect_no_crash': True,
    })

    test_cases.append({
        'id': 'edge_garbage',
        'query': 'xyzzy frobnicator quantum banana',
        'category': 'edge_case',
        'expected_relevant': {},
        'min_hit_rate_at_10': 0.0,
        'description': 'Garbage query should return gracefully (recent nodes)',
        'expect_no_crash': True,
    })

    # 5b: Very long query
    test_cases.append({
        'id': 'edge_long_query',
        'query': 'What is the current state of the budget screen layout and how does it relate to the tier pricing and spend mode toggle and what decisions were made about the horizontal scrollable design',
        'category': 'edge_case',
        'expected_relevant': {},  # Will fill from actual locked budget nodes
        'min_hit_rate_at_10': 0.0,
        'description': 'Long verbose query should still work',
        'expect_no_crash': True,
    })

    # Find budget-related nodes for the long query
    budget_nodes = conn.execute('''
        SELECT id FROM nodes
        WHERE (title LIKE '%budget%' OR keywords LIKE '%budget%')
        AND locked = 1
        LIMIT 5
    ''').fetchall()
    if budget_nodes:
        test_cases[-1]['expected_relevant'] = {n['id']: 1 for n in budget_nodes}
        test_cases[-1]['min_hit_rate_at_10'] = 0.5

    # 5c: Single word query
    test_cases.append({
        'id': 'edge_single_word',
        'query': 'budget',
        'category': 'edge_case',
        'expected_relevant': {n['id']: 1 for n in budget_nodes} if budget_nodes else {},
        'min_hit_rate_at_10': 0.5 if budget_nodes else 0.0,
        'description': 'Single-word query should recall relevant nodes',
    })

    # ═══════════════════════════════════════════════════════════
    # CATEGORY 6: Cross-Domain Recall
    # Queries that span multiple node types / projects
    # ═══════════════════════════════════════════════════════════

    projects = conn.execute('''
        SELECT id, title FROM nodes WHERE type = 'project' AND locked = 1
        ORDER BY access_count DESC LIMIT 3
    ''').fetchall()

    for proj in projects:
        title = proj['title'] or ''
        topic = _extract_topic(title)
        if not topic:
            continue

        # Project's children via edges
        children = conn.execute('''
            SELECT target_id FROM edges
            WHERE source_id = ? AND edge_type = 'part_of'
            LIMIT 10
        ''', (proj['id'],)).fetchall()

        relevance = {proj['id']: 2}
        for c in children:
            relevance[c[0]] = 1

        test_cases.append({
            'id': f'project_{proj["id"][:8]}',
            'query': f'everything about {topic}',
            'category': 'cross_domain',
            'expected_relevant': relevance,
            'min_hit_rate_at_10': 1.0,
            'description': f'Cross-domain project recall: {title[:60]}',
        })

    conn.close()

    # Deduplicate by ID
    seen = set()
    unique_cases = []
    for tc in test_cases:
        if tc['id'] not in seen:
            seen.add(tc['id'])
            unique_cases.append(tc)

    # Save if output path given
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(unique_cases, f, indent=2)
        print(f'[golden] Generated {len(unique_cases)} test cases → {output_path}')

    return unique_cases


def _extract_query_from_title(title: str, node_type: str) -> str:
    """Extract a natural search query from a node title."""
    if not title:
        return ''

    # Strip common prefixes
    cleaned = re.sub(r'^\[.*?\]\s*', '', title)  # [o_glo] prefix
    cleaned = re.sub(r'^(Rule|Decision|Correction|Procedure|Context):\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^(Rejected|Paused|Updated):\s*', '', cleaned, flags=re.IGNORECASE)

    # Take the meaningful part (first clause, up to dash or period)
    parts = re.split(r'\s*[—–\.\n]', cleaned)
    query = parts[0].strip() if parts else cleaned.strip()

    # Truncate long queries
    words = query.split()
    if len(words) > 12:
        query = ' '.join(words[:10])

    return query.lower().strip()


def _extract_topic(title: str) -> str:
    """Extract a short topic phrase from a title for use in intent queries."""
    if not title:
        return ''

    cleaned = re.sub(r'^\[.*?\]\s*', '', title)
    cleaned = re.sub(r'^(Rule|Decision|Correction|Procedure|Context):\s*', '', cleaned, flags=re.IGNORECASE)

    # Take first meaningful phrase
    parts = re.split(r'\s*[—–\.\n:,]', cleaned)
    topic = parts[0].strip() if parts else ''

    words = topic.split()
    if len(words) > 8:
        topic = ' '.join(words[:6])

    return topic.lower().strip()


if __name__ == '__main__':
    # Auto-discover brain.db
    search_paths = [
        os.path.expanduser('~/Documents/Claude/AgentsContext/tmemory/brain.db'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'brain.db'),
        os.environ.get('TMEMORY_DB_DIR', ''),
    ]

    db_path = None
    for p in search_paths:
        if p and os.path.exists(p):
            db_path = p
            break

    if not db_path:
        if len(sys.argv) > 1:
            db_path = sys.argv[1]
        else:
            print('Usage: python generate_golden.py [brain.db path]')
            sys.exit(1)

    output = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
    cases = generate_golden_dataset(db_path, output)
    print(f'\nGenerated {len(cases)} test cases across categories:')
    categories = {}
    for c in cases:
        cat = c['category']
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f'  {cat}: {count}')
