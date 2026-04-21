#!/usr/bin/env python3
"""Export brain graph as training data in three formats.

Generates training JSONL for A/B testing different traversal strategies
on the same node set. All formats use the same 100 nodes from 8 communities.

Format B: Typed edge walks — follow one edge type at a time (depth 1-2)
Format C: Top-down community — narrative → members → internal edges
Format C2: Bottom-up fractal — node → edges → discover community → peers

Usage:
    python3 eval/export_training_data.py
    python3 eval/export_training_data.py --output-dir /tmp/brain_training
"""

import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# 8 communities selected for diversity
COMMUNITY_IDS = [
    'e64a8d82',  # Fractal Architecture: Gaps to S2 (S, 27, dense)
    '119376dc',  # Fractal O/K/Δ: Self-Similarity (A, 24)
    '1528a969',  # Anchor's Identity and Growth (A, 23, sparse)
    'a0a98408',  # Identity vs Instructions (S, 13)
    'aa96adc9',  # Encoder V3: Regression to Contract (S, 15, corrections)
    '0f725b5b',  # Tom's Evaluation Framework (S, 12)
    'dc4416ef',  # S2CD Quality Control (F, 3, tiny)
    'fab37417',  # Hub Dominance: Fatigue Solution (C, 13)
]

NOISE_RELATIONS = {'community_member', 'co_accessed', 'emergent_bridge'}


def load_graph(brain):
    """Load all nodes, edges, metadata for our test set."""
    # Resolve community IDs and get members
    communities = {}
    all_node_ids = set()

    for cid_short in COMMUNITY_IDS:
        row = brain.conn.execute(
            'SELECT id, title, content FROM nodes WHERE id LIKE ?',
            (cid_short + '%',)).fetchone()
        if not row:
            print('WARNING: community %s not found' % cid_short)
            continue

        cid = row[0]
        members = brain.conn.execute('''
            SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            FROM edges e
            JOIN edge_relations er ON er.edge_id = e.edge_id
            WHERE (e.source_id = ? OR e.target_id = ?)
            AND er.relation = 'community_member'
            AND er.archived = 0
        ''', (cid, cid, cid)).fetchall()

        member_ids = [m[0] for m in members]
        meta = dict(brain.conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
            (cid,)).fetchall())

        communities[cid_short] = {
            'id': cid, 'title': row[1], 'content': row[2],
            'members': member_ids,
            'narrative': meta.get('community_narrative', ''),
            'maturity': meta.get('community_maturity', '?'),
            'open_questions': meta.get('community_open_questions', ''),
            'size': meta.get('community_size', str(len(member_ids))),
        }
        all_node_ids.update(member_ids)

    # Load full node data
    nodes = {}
    for nid in all_node_ids:
        row = brain.conn.execute(
            'SELECT id, type, title, content, confidence, encoding_source, '
            'locked, critical, keywords, created_at, updated_at '
            'FROM nodes WHERE id = ?', (nid,)).fetchone()
        if not row:
            continue
        meta = dict(brain.conn.execute(
            'SELECT key, value FROM node_metadata_kv WHERE node_id = ?',
            (nid,)).fetchall())
        nodes[nid] = {
            'id': nid, 'type': row[1], 'title': row[2],
            'content': row[3] or '', 'confidence': row[4],
            'encoding_source': row[5] or '', 'locked': bool(row[6]),
            'critical': bool(row[7]), 'keywords': row[8] or '',
            'created_at': row[9] or '', 'updated_at': row[10] or '',
            **{k: v for k, v in meta.items() if k != 'revision_history'},
        }

    # Load edges between our nodes
    nl = list(all_node_ids)
    ph = ','.join('?' * len(nl))
    edge_rows = brain.conn.execute('''
        SELECT e.source_id, e.target_id, er.relation, er.description
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE e.source_id IN (%s) AND e.target_id IN (%s)
        AND er.relation NOT IN (%s)
        AND er.archived = 0
    ''' % (ph, ph, ','.join('?' * len(NOISE_RELATIONS))),
        nl * 2 + list(NOISE_RELATIONS)).fetchall()

    edges = []
    edge_map = defaultdict(list)  # source → [(target, relation, description)]
    for s, t, r, d in edge_rows:
        edges.append({'source': s, 'target': t, 'relation': r, 'description': d or ''})
        edge_map[s].append((t, r, d or ''))

    # Map node → community
    node_to_community = {}
    for cid_short, comm in communities.items():
        for mid in comm['members']:
            node_to_community.setdefault(mid, []).append(cid_short)

    return communities, nodes, edges, edge_map, node_to_community


def format_node(n):
    """Render a node with all its fields."""
    lines = ['[%s] "%s" (id:%s)' % (n['type'], n['title'], n['id'][:8])]
    lines.append('Confidence: %.2f | Source: %s | Locked: %s' % (
        n['confidence'], n['encoding_source'][:20], n['locked']))
    if n.get('created_at'):
        lines.append('Created: %s' % n['created_at'][:10])
    if n['content']:
        lines.append('Content: %s' % n['content'][:600])
    # All KV metadata
    skip = {'type', 'title', 'content', 'confidence', 'encoding_source',
            'locked', 'critical', 'keywords', 'created_at', 'updated_at', 'id'}
    for k, v in n.items():
        if k not in skip and v and str(v).strip():
            lines.append('%s: %s' % (k, str(v)[:200]))
    return '\n'.join(lines)


def format_edge(source_node, target_node, relation, description):
    """Render an edge between two nodes."""
    lines = ['[%s] "%s" (id:%s)' % (source_node['type'], source_node['title'], source_node['id'][:8])]
    lines.append('  %s →' % relation)
    if description:
        lines.append('  (%s)' % description[:150])
    lines.append('[%s] "%s" (id:%s)' % (target_node['type'], target_node['title'], target_node['id'][:8]))
    return '\n'.join(lines)


def generate_format_b(nodes, edges, edge_map):
    """Format B: Typed edge walks — depth 1 + depth 2 chains."""
    examples = []

    # Depth 1: every edge
    for e in edges:
        src = nodes.get(e['source'])
        tgt = nodes.get(e['target'])
        if not src or not tgt:
            continue

        text = '[EDGE: %s]\n%s\n  %s →\n' % (e['relation'], format_node(src), e['relation'])
        if e['description']:
            text += '  (%s)\n' % e['description'][:150]
        text += format_node(tgt)
        examples.append({'text': text, 'format': 'B', 'depth': 1,
                         'edge_type': e['relation']})

    # Depth 2: chain where A→B→C follows through
    for e in edges:
        src = nodes.get(e['source'])
        mid = nodes.get(e['target'])
        if not src or not mid:
            continue
        for t2, r2, d2 in edge_map.get(e['target'], []):
            if t2 == e['source']:
                continue  # no backtrack
            tgt = nodes.get(t2)
            if not tgt:
                continue

            text = '[CHAIN: %s → %s]\n' % (e['relation'], r2)
            text += format_node(src) + '\n'
            text += '  %s →\n' % e['relation']
            if e['description']:
                text += '  (%s)\n' % e['description'][:100]
            text += format_node(mid) + '\n'
            text += '  %s →\n' % r2
            if d2:
                text += '  (%s)\n' % d2[:100]
            text += format_node(tgt)
            examples.append({'text': text, 'format': 'B', 'depth': 2,
                             'edge_type': '%s→%s' % (e['relation'], r2)})

    return examples


def generate_format_c(communities, nodes, edges, edge_map, node_to_community):
    """Format C: Top-down community — traverse every edge, framed by community context.

    Every edge is visited, organized by community. For each community:
    1. Community narrative as framing context
    2. Each internal edge as a traversal: community → source member → edge → target member
    3. Cross-community edges: community → member → edge → external node → their community
    """
    examples = []
    _node_to_community = node_to_community

    # Map edges to communities
    for cid_short, comm in communities.items():
        member_set = set(comm['members'])
        member_nodes = {mid: nodes[mid] for mid in comm['members'] if mid in nodes}
        internal = [e for e in edges
                    if e['source'] in member_set and e['target'] in member_set]
        outgoing = [e for e in edges
                    if e['source'] in member_set and e['target'] not in member_set]

        comm_header = '[COMMUNITY: %s]\n' % comm['title']
        comm_header += 'Maturity: %s | Size: %s\n' % (comm['maturity'], comm['size'])
        if comm['narrative']:
            comm_header += 'Narrative: %s\n' % comm['narrative'][:300]

        # Community overview with all members
        text = comm_header + '\nMembers:\n'
        for mn in member_nodes.values():
            text += '  [%s] "%s" (id:%s, conf:%.2f)\n' % (
                mn['type'], mn['title'][:50], mn['id'][:8], mn['confidence'])
        examples.append({'text': text, 'format': 'C', 'type': 'overview',
                         'community': cid_short})

        # Each internal edge: community context → source → edge → target
        for e in internal:
            src = nodes.get(e['source'])
            tgt = nodes.get(e['target'])
            if not src or not tgt:
                continue
            text = comm_header + '\n[INTERNAL EDGE]\n'
            text += format_node(src) + '\n'
            text += '  %s →\n' % e['relation']
            if e['description']:
                text += '  (%s)\n' % e['description'][:150]
            text += format_node(tgt)
            examples.append({'text': text, 'format': 'C', 'type': 'internal_edge',
                             'community': cid_short, 'edge_type': e['relation']})

        # Each outgoing edge: community → member → edge → external → their community
        for e in outgoing:
            src = nodes.get(e['source'])
            tgt = nodes.get(e['target'])
            if not src or not tgt:
                continue
            text = comm_header + '\n[CROSS-COMMUNITY EDGE]\n'
            text += format_node(src) + '\n'
            text += '  %s → (leaving community)\n' % e['relation']
            if e['description']:
                text += '  (%s)\n' % e['description'][:150]
            text += format_node(tgt)
            # What community does the target belong to?
            tgt_comms = _node_to_community.get(e['target'], [])
            for tc in tgt_comms:
                tc_data = communities.get(tc, {})
                if tc_data:
                    text += '\n→ Arrives in community: %s (%s)' % (
                        tc_data.get('title', '?')[:50], tc_data.get('maturity', '?'))
            examples.append({'text': text, 'format': 'C', 'type': 'cross_edge',
                             'community': cid_short, 'edge_type': e['relation']})

        # Depth 2: internal chains within community
        for e in internal:
            for t2, r2, d2 in edge_map.get(e['target'], []):
                if t2 == e['source'] or t2 not in member_set:
                    continue
                tgt2 = nodes.get(t2)
                if not tgt2:
                    continue
                src = nodes.get(e['source'])
                mid = nodes.get(e['target'])
                if not src or not mid:
                    continue
                text = comm_header + '\n[INTERNAL CHAIN]\n'
                text += format_node(src) + '\n'
                text += '  %s →\n' % e['relation']
                text += format_node(mid) + '\n'
                text += '  %s →\n' % r2
                text += format_node(tgt2)
                examples.append({'text': text, 'format': 'C', 'type': 'internal_chain',
                                 'community': cid_short})

    return examples


def generate_format_c2(communities, nodes, edges, edge_map, node_to_community):
    """Format C2: Bottom-up fractal — traverse every edge, framed by starting node's perspective.

    Every edge is visited (same as B and C). For each edge:
    1. Start from source node with full context
    2. Follow edge to target
    3. Discover what community the target belongs to
    4. See peers in that community

    Plus depth 2 chains following the same bottom-up pattern.
    """
    examples = []

    # Every edge as a bottom-up walk: node → edge → arrive → discover community
    for e in edges:
        src = nodes.get(e['source'])
        tgt = nodes.get(e['target'])
        if not src or not tgt:
            continue

        text = '[STARTING FROM NODE]\n'
        text += format_node(src) + '\n'

        # Source's community context
        src_comms = node_to_community.get(e['source'], [])
        if src_comms:
            sc = communities.get(src_comms[0], {})
            text += 'My community: %s (%s)\n' % (
                sc.get('title', '?')[:50], sc.get('maturity', '?'))

        text += '\n[FOLLOWING EDGE: %s]\n' % e['relation']
        if e['description']:
            text += '(%s)\n' % e['description'][:150]

        text += '\n[ARRIVING AT]\n'
        text += format_node(tgt) + '\n'

        # Discover target's community
        tgt_comms = node_to_community.get(e['target'], [])
        if tgt_comms:
            for tc in tgt_comms:
                comm = communities.get(tc, {})
                same = 'SAME' if tc in src_comms else 'DIFFERENT'
                text += '\n[DISCOVERED: %s community — %s]\n' % (
                    same, comm.get('title', '?')[:50])
                if comm.get('narrative'):
                    text += '%s\n' % comm['narrative'][:200]
                # Peers
                peers = [mid for mid in comm.get('members', [])
                         if mid != e['target'] and mid != e['source'] and mid in nodes]
                if peers:
                    text += 'Other members here:\n'
                    for pid in peers[:5]:
                        p = nodes[pid]
                        text += '  [%s] "%s" (id:%s)\n' % (
                            p['type'], p['title'][:50], pid[:8])

        examples.append({'text': text, 'format': 'C2', 'type': 'edge_walk',
                         'edge_type': e['relation']})

    # Depth 2: node → edge → node → edge → node (bottom-up discovery chain)
    for e in edges:
        src = nodes.get(e['source'])
        mid = nodes.get(e['target'])
        if not src or not mid:
            continue
        for t2, r2, d2 in edge_map.get(e['target'], []):
            if t2 == e['source']:
                continue
            tgt = nodes.get(t2)
            if not tgt:
                continue

            text = '[FRACTAL CHAIN — building up]\n'
            text += '[START] ' + format_node(src) + '\n'
            text += '  %s →\n' % e['relation']
            if e['description']:
                text += '  (%s)\n' % e['description'][:100]
            text += '[HOP 1] ' + format_node(mid) + '\n'
            text += '  %s →\n' % r2
            if d2:
                text += '  (%s)\n' % d2[:100]
            text += '[HOP 2] ' + format_node(tgt) + '\n'

            # What community did we end up in?
            end_comms = node_to_community.get(t2, [])
            start_comms = node_to_community.get(e['source'], [])
            if end_comms:
                tc = end_comms[0]
                comm = communities.get(tc, {})
                same = 'SAME' if tc in start_comms else 'DIFFERENT'
                text += '\n[ARRIVED IN: %s community — %s]\n' % (
                    same, comm.get('title', '?')[:50])

            examples.append({'text': text, 'format': 'C2', 'type': 'fractal_chain',
                             'edge_type': '%s→%s' % (e['relation'], r2)})

    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/tmp/brain_training')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    os.environ.setdefault('BRAIN_DB_DIR',
                          os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain'))
    from servers.brain import Brain
    brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))

    print('Loading graph...')
    communities, nodes, edges, edge_map, node_to_community = load_graph(brain)
    print('  %d communities, %d nodes, %d edges' % (
        len(communities), len(nodes), len(edges)))

    # Generate all three formats
    print('\nGenerating Format B (typed edge walks)...')
    b_examples = generate_format_b(nodes, edges, edge_map)
    print('  %d examples' % len(b_examples))

    print('Generating Format C (top-down community)...')
    c_examples = generate_format_c(communities, nodes, edges, edge_map, node_to_community)
    print('  %d examples' % len(c_examples))

    print('Generating Format C2 (bottom-up fractal)...')
    c2_examples = generate_format_c2(communities, nodes, edges, edge_map, node_to_community)
    print('  %d examples' % len(c2_examples))

    # Write JSONL files
    for name, examples in [('B_typed_walks', b_examples),
                           ('C_topdown', c_examples),
                           ('C2_bottomup', c2_examples)]:
        path = os.path.join(args.output_dir, '%s.jsonl' % name)
        with open(path, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        total_chars = sum(len(ex['text']) for ex in examples)
        print('\n%s: %d examples, %dK chars (~%dK tokens) → %s' % (
            name, len(examples), total_chars // 1000, total_chars // 4000, path))

    # Stats summary
    print('\n=== SUMMARY ===')
    print('Nodes: %d | Edges: %d | Communities: %d' % (
        len(nodes), len(edges), len(communities)))
    print('B: %d examples | C: %d examples | C2: %d examples' % (
        len(b_examples), len(c_examples), len(c2_examples)))
    print('Output: %s' % args.output_dir)


if __name__ == '__main__':
    main()
