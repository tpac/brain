#!/usr/bin/env python3
"""S2 Community Detection Eval — measures decoder + encoder quality.

Runs against a brain copy. Produces a scored report across multiple dimensions.
Use for: baseline measurement, A/B testing prompts, regression detection.

Usage:
    python3 eval/s2_community_eval.py                    # Full eval
    python3 eval/s2_community_eval.py --db /path/to/copy  # Against specific DB
    python3 eval/s2_community_eval.py --report /path/to/report.json  # Score existing
"""

import json
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime


def score_community(conn, community_id):
    """Score a single community node across quality dimensions.

    Returns dict with dimension scores (0-1) and details.
    """
    node = conn.execute(
        "SELECT id, title, content, keywords, confidence, encoding_source "
        "FROM nodes WHERE id = ?", (community_id,)).fetchone()

    if not node:
        return {'score': 0, 'error': 'not found'}

    nid, title, content, keywords, conf, enc_src = node
    scores = {}
    details = {}

    # ── 1. Content quality (0-1) ──
    content_len = len(content or '')
    if content_len >= 300:
        scores['content_length'] = 1.0
    elif content_len >= 150:
        scores['content_length'] = 0.7
    elif content_len >= 50:
        scores['content_length'] = 0.4
    else:
        scores['content_length'] = 0.1
    details['content_length'] = content_len

    # Does content reference node IDs? (narrative with references)
    import re
    id_refs = re.findall(r'id:[a-z0-9_]{6,8}', content or '')
    scores['content_references'] = min(1.0, len(id_refs) / 3)
    details['id_references'] = len(id_refs)

    # ── 2. Situation quality (0-1) ──
    sit = conn.execute(
        "SELECT value FROM node_metadata_kv WHERE node_id = ? AND key = 'situation'",
        (nid,)).fetchone()
    sit_text = sit[0] if sit and sit[0] else ''
    if len(sit_text) >= 50:
        scores['situation'] = 1.0
    elif len(sit_text) >= 20:
        scores['situation'] = 0.5
    else:
        scores['situation'] = 0.0
    details['situation_length'] = len(sit_text)

    # ── 3. Metadata completeness (0-1) ──
    required_keys = {
        'community_narrative', 'community_key_decisions',
        'community_maturity', 'community_dominant_type',
        'community_members',
    }
    optional_keys = {
        'community_open_questions', 'community_latest_development',
        'community_learning_arc',
    }

    meta = dict(conn.execute(
        "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
        (nid,)).fetchall())

    required_present = sum(1 for k in required_keys
                           if k in meta and meta[k])
    scores['required_metadata'] = required_present / len(required_keys)
    details['required_present'] = required_present
    details['required_total'] = len(required_keys)

    optional_present = sum(1 for k in optional_keys
                           if k in meta and meta[k])
    scores['optional_metadata'] = optional_present / len(optional_keys)
    details['optional_present'] = optional_present

    # ── 4. Key decisions quality ──
    key_dec = meta.get('community_key_decisions', '')
    if ':' in key_dec:
        # Has "id: title" format
        scores['key_decisions_format'] = 1.0
    elif key_dec:
        scores['key_decisions_format'] = 0.5  # IDs only
    else:
        scores['key_decisions_format'] = 0.0
    details['key_decisions'] = key_dec[:100]

    # ── 5. Members quality ──
    members_meta = meta.get('community_members', '')
    member_edges = conn.execute("""
        SELECT COUNT(*) FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE (e.source_id = ? OR e.target_id = ?)
        AND er.relation = 'community_member'
        AND er.archived = 0
    """, (nid, nid)).fetchone()[0]

    scores['member_edges'] = min(1.0, member_edges / 3)
    details['member_edges'] = member_edges

    if ':' in members_meta:
        scores['members_format'] = 1.0
    elif members_meta:
        scores['members_format'] = 0.5
    else:
        scores['members_format'] = 0.0

    # ── 6. Open questions (0-1) ──
    open_q = meta.get('community_open_questions', '')
    if open_q and len(open_q) > 5 and open_q.lower() != 'none':
        scores['open_questions'] = 1.0
    else:
        scores['open_questions'] = 0.0
    details['open_questions'] = open_q[:100]

    # ── 7. Title quality ──
    if ':' in (title or ''):
        scores['title_narrative'] = 1.0  # Has "X: From Y to Z" format
    elif len(title or '') > 20:
        scores['title_narrative'] = 0.7
    else:
        scores['title_narrative'] = 0.3
    details['title'] = title

    # ── Composite ──
    weights = {
        'content_length': 0.15,
        'content_references': 0.10,
        'situation': 0.15,
        'required_metadata': 0.15,
        'optional_metadata': 0.05,
        'key_decisions_format': 0.10,
        'member_edges': 0.10,
        'members_format': 0.05,
        'open_questions': 0.10,
        'title_narrative': 0.05,
    }

    composite = sum(scores.get(k, 0) * w for k, w in weights.items())

    return {
        'id': nid,
        'title': title,
        'confidence': conf,
        'composite_score': round(composite, 3),
        'dimension_scores': scores,
        'details': details,
    }


def score_decoder(decode_stats):
    """Score decoder output quality."""
    scores = {}

    clusters = decode_stats.get('valid_clusters', 0)
    scores['cluster_count'] = min(1.0, clusters / 10) if clusters > 0 else 0

    fragments = decode_stats.get('fragments_dissolved', 0)
    total_seeded = decode_stats.get('clusters_seeded', 0)
    if total_seeded > 0:
        scores['fragment_rate'] = 1.0 - (fragments / total_seeded)
    else:
        scores['fragment_rate'] = 0

    corridors = decode_stats.get('corridors', 0)
    if clusters > 0:
        corridor_rate = corridors / clusters
        # Some corridors expected — 10-30% is healthy
        if 0.1 <= corridor_rate <= 0.3:
            scores['corridor_health'] = 1.0
        elif corridor_rate < 0.1:
            scores['corridor_health'] = 0.7  # Maybe too aggressive
        else:
            scores['corridor_health'] = 0.5  # Too many corridors
    else:
        scores['corridor_health'] = 0

    return scores


def run_eval(db_path=None):
    """Run full S2C eval. Returns report dict."""
    if db_path is None:
        db_path = os.path.join(
            os.path.expanduser('~/AgentsContext/brain'), 'brain.db')

    conn = sqlite3.connect(db_path, timeout=5)

    # Find all S2CE community nodes
    communities = conn.execute("""
        SELECT id, title, confidence FROM nodes
        WHERE type = 'community' AND archived = 0
        AND encoding_source = 's2:community_detection'
        ORDER BY confidence DESC
    """).fetchall()

    if not communities:
        print("No S2CE communities found")
        conn.close()
        return {'communities': 0}

    # Score each
    results = []
    for cid, title, conf in communities:
        result = score_community(conn, cid)
        results.append(result)

    # Aggregate
    composites = [r['composite_score'] for r in results]
    dimension_agg = {}
    for dim in results[0]['dimension_scores']:
        values = [r['dimension_scores'].get(dim, 0) for r in results]
        dimension_agg[dim] = {
            'mean': round(sum(values) / len(values), 3),
            'min': round(min(values), 3),
            'max': round(max(values), 3),
        }

    # Duplicate detection
    titles = [r['title'] for r in results]
    title_words = [set(t.lower().split()[:3]) for t in titles]
    potential_dupes = 0
    for i in range(len(title_words)):
        for j in range(i + 1, len(title_words)):
            if len(title_words[i] & title_words[j]) >= 2:
                potential_dupes += 1

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'communities': len(results),
        'composite_score': {
            'mean': round(sum(composites) / len(composites), 3),
            'min': round(min(composites), 3),
            'max': round(max(composites), 3),
        },
        'dimensions': dimension_agg,
        'potential_duplicates': potential_dupes,
        'per_community': results,
    }

    conn.close()
    return report


def print_report(report):
    """Pretty-print eval report."""
    print("=" * 70)
    print("S2C COMMUNITY EVAL REPORT")
    print("=" * 70)

    print("\nCommunities: %d" % report['communities'])
    cs = report['composite_score']
    print("Composite score: mean=%.3f  min=%.3f  max=%.3f" % (
        cs['mean'], cs['min'], cs['max']))
    print("Potential duplicates: %d" % report.get('potential_duplicates', 0))

    print("\nDimension breakdown:")
    for dim, stats in sorted(report['dimensions'].items()):
        bar = '█' * int(stats['mean'] * 20)
        print("  %-25s %.2f  %s" % (dim, stats['mean'], bar))

    # Bottom 5
    bottom = sorted(report['per_community'],
                    key=lambda r: r['composite_score'])[:5]
    print("\nWeakest communities:")
    for r in bottom:
        print("  [%.2f] %s" % (r['composite_score'], r['title'][:55]))
        weak_dims = [d for d, s in r['dimension_scores'].items() if s < 0.5]
        if weak_dims:
            print("         weak: %s" % ', '.join(weak_dims))

    # Top 5
    top = sorted(report['per_community'],
                 key=lambda r: -r['composite_score'])[:5]
    print("\nStrongest communities:")
    for r in top:
        print("  [%.2f] %s" % (r['composite_score'], r['title'][:55]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', help='Path to brain.db')
    parser.add_argument('--report', help='Score existing report JSON')
    parser.add_argument('--save', help='Save report to JSON file')
    args = parser.parse_args()

    if args.report:
        with open(args.report) as f:
            report = json.load(f)
    else:
        report = run_eval(args.db)

    print_report(report)

    if args.save:
        with open(args.save, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print("\nSaved to %s" % args.save)
