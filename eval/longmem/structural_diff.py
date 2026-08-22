"""Structural encoding comparison across encoder versions.

Sidesteps the pass/fail verdict (which depends on the noisy recall+
surface+answerer pipeline) and asks: for each eval item, what did the
encoder actually WRITE? Did the dated events get event_time? Did the
edges connect events temporally? How does v15.8 vs v15.9 vs v15.10
compare on:

  - Node count + event_time emission rate
  - Edge density (per node) + temporal-relation edges (before / after /
    meets / met_by / during / anchored_to)
  - my_raw_quote / their_raw_quote presence (voice symmetry signal)
  - open-type + correction-type nodes (live-contradiction / explicit
    paraphrase rejection — v15.9 specifically wrote "correction" nodes
    calling out assistant hallucinations)
  - For each named event in the gold question, does it get encoded
    with a non-None event_time?

USE
    ./dev python3 eval/longmem/structural_diff.py \\
        temporal_v15_8_2026_05_13_214239 \\
        temporal_v15_9_2026_05_13_214239 \\
        temporal_v15_10_2026_05_14_HHMMSS \\
        --out eval/longmem/reports/v15_10_compare_2026_05_14/structural_diff.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from eval.longmem.artifacts import load_artifacts, list_items


# ─── Temporal relation aspects (what counts as a temporal edge) ────────────

TEMPORAL_RELATIONS = {
    'before', 'after', 'meets', 'met_by', 'during',
    'anchored_to', 'precedes', 'follows',
}


def _nodes_with_kv(nodes, kv_key):
    return sum(1 for n in nodes if (n.get('kv') or {}).get(kv_key))


def _nodes_by_type(nodes, type_name):
    return sum(1 for n in nodes if n.get('type') == type_name)


def _node_aspects(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Per-item structural summary — encoder behavior only, no recall/surface."""
    nodes = bundle.get('nodes') or []
    edges = bundle.get('edges') or []
    nc = len(nodes)
    return {
        'node_count': nc,
        'event_time_count': _nodes_with_kv(nodes, 'event_time'),
        'event_time_rate': _nodes_with_kv(nodes, 'event_time') / nc if nc else 0,
        'their_raw_quote': _nodes_with_kv(nodes, 'their_raw_quote'),
        'my_raw_quote': _nodes_with_kv(nodes, 'my_raw_quote'),
        'reasoning': _nodes_with_kv(nodes, 'reasoning'),
        'keywords': _nodes_with_kv(nodes, 'keywords'),
        'situation': _nodes_with_kv(nodes, 'situation'),
        # Type breakdown
        'type_event': _nodes_by_type(nodes, 'event'),
        'type_decision': _nodes_by_type(nodes, 'decision'),
        'type_moment': _nodes_by_type(nodes, 'moment'),
        'type_fact': _nodes_by_type(nodes, 'fact'),
        'type_open': _nodes_by_type(nodes, 'open'),
        'type_correction': _nodes_by_type(nodes, 'correction'),
        'type_time_anchor': _nodes_by_type(nodes, 'time_anchor'),
        # Edge structure
        'edge_count': len(edges),
        'edges_per_node': len(edges) / nc if nc else 0,
        'temporal_edges': sum(1 for e in edges
                              if e.get('relation') in TEMPORAL_RELATIONS),
    }


def _dated_events(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """For per-item inspection — every node with event_time set, in
    chronological order, with the brief shape Tom can scan."""
    out = []
    for n in bundle.get('nodes') or []:
        kv = n.get('kv') or {}
        et = kv.get('event_time')
        if not et:
            continue
        out.append({
            'id': (n.get('id') or '')[:8],
            'type': n.get('type'),
            'title': (n.get('title') or '')[:90],
            'event_time': et,
            'user_quote': bool(kv.get('their_raw_quote')),
            'anchor_quote': bool(kv.get('my_raw_quote')),
        })
    out.sort(key=lambda r: r['event_time'])
    return out


def _gold_anchor_hit(bundle: Dict[str, Any], gold_keywords: List[str]) -> Optional[Dict[str, Any]]:
    """Find a node whose title or content matches gold_keywords AND has event_time.
    Returns the node summary or None."""
    if not gold_keywords:
        return None
    keys = [k.lower() for k in gold_keywords]
    for n in bundle.get('nodes') or []:
        kv = n.get('kv') or {}
        et = kv.get('event_time')
        if not et:
            continue
        haystack = ((n.get('title') or '') + ' ' + (n.get('content') or '')).lower()
        if all(k in haystack for k in keys):
            return {
                'id': (n.get('id') or '')[:8],
                'type': n.get('type'),
                'title': (n.get('title') or '')[:90],
                'event_time': et,
            }
    return None


# Per-item "what's the gold-required dated event" — keywords that should
# appear in the title/content of an event-bearing node, the expected
# event_time, and a short label.
ITEM_PROBES: Dict[str, Dict[str, Any]] = {
    'gpt4_85da3956': {
        'label': 'Universal Studios visit',
        'gold_keywords': ['universal'],
        'expected_event_time': '2023-07-15',
    },
    'gpt4_b0863698': {
        'label': '5K charity run',
        'gold_keywords': ['5k'],
        'expected_event_time': '2023-03-19',
    },
    '982b5123': {
        'label': 'Wedding (Mar 2023)',
        'gold_keywords': ['wedding'],
        'expected_event_time': '2023-03',  # prefix match
    },
    '71017276': {
        'label': 'Chandelier receipt',
        'gold_keywords': ['chandelier'],
        'expected_event_time': '2023-03-04',
    },
    'gpt4_4edbafa2': {
        'label': 'First BBQ in June',
        'gold_keywords': ['bbq'],
        'expected_event_time': '2023-06',
    },
    'gpt4_2487a7cb': {
        'label': 'Time Management workshop',
        'gold_keywords': ['time management'],
        'expected_event_time': None,  # any date — ordering question
    },
    '0bb5a684': {
        'label': 'Workshop on Jan 10 + meeting Jan 17',
        'gold_keywords': ['workshop'],
        'expected_event_time': '2023-01-10',
    },
    'e4e14d04': {
        'label': 'Book Lovers Unite meetup',
        'gold_keywords': ['meetup'],
        'expected_event_time': None,
    },
    '08f4fc43': {
        'label': 'Sunday mass / Ash Wednesday',
        'gold_keywords': ['ash wednesday'],
        'expected_event_time': None,
    },
    '982b5123_abs': {
        'label': '(abstention)',
        'gold_keywords': [],
        'expected_event_time': None,
    },
    'gpt4_93159ced_abs': {
        'label': '(abstention)',
        'gold_keywords': [],
        'expected_event_time': None,
    },
    'gpt4_70e84552_abs': {
        'label': '(abstention)',
        'gold_keywords': [],
        'expected_event_time': None,
    },
}


def render_report(runs: List[str], items_by_qid: Dict[str, Dict[str, Dict[str, Any]]],
                   labels: List[str]) -> str:
    """runs: list of run_names. items_by_qid: {qid: {run_name: bundle}}."""
    out = []
    out.append(f"# Structural encoding comparison — {' vs '.join(labels)}")
    out.append('')
    out.append('No pass/fail verdicts. Pure encoder-output structure.')
    out.append('')

    # ─── Cohort-aggregate table ─────────────────────────────────────
    out.append('## Cohort-aggregate signals')
    out.append('')
    cohort_aspects = []
    for run in runs:
        agg = Counter()
        for qid, by_run in items_by_qid.items():
            b = by_run.get(run) or {}
            if not b:
                continue
            asp = _node_aspects(b)
            for k, v in asp.items():
                if isinstance(v, (int, float)):
                    agg[k] += v
        cohort_aspects.append(agg)

    metrics = [
        ('node_count', 'Total nodes'),
        ('event_time_count', 'Nodes with event_time'),
        ('their_raw_quote', 'Nodes with their_raw_quote'),
        ('my_raw_quote', 'Nodes with my_raw_quote'),
        ('reasoning', 'Nodes with reasoning kv'),
        ('keywords', 'Nodes with keywords'),
        ('situation', 'Nodes with situation'),
        ('type_event', 'event-type nodes'),
        ('type_decision', 'decision-type nodes'),
        ('type_moment', 'moment-type nodes'),
        ('type_fact', 'fact-type nodes'),
        ('type_open', 'open-type nodes (live-contradiction)'),
        ('type_correction', 'correction-type nodes'),
        ('type_time_anchor', 'time_anchor-type nodes'),
        ('edge_count', 'Total edges'),
        ('temporal_edges', 'Temporal-relation edges'),
    ]
    header = '| Signal | ' + ' | '.join(labels) + ' |'
    sep = '|---|' + '---:|' * len(labels)
    out.append(header)
    out.append(sep)
    for k, label in metrics:
        row = [label] + [str(cohort_aspects[i].get(k, 0)) for i in range(len(runs))]
        out.append('| ' + ' | '.join(row) + ' |')
    # event_time rate
    rate_row = ['event_time / total (rate)']
    for i in range(len(runs)):
        nc = cohort_aspects[i].get('node_count', 0)
        et = cohort_aspects[i].get('event_time_count', 0)
        rate_row.append(f'{(et/nc*100) if nc else 0:.0f}%')
    out.append('| ' + ' | '.join(rate_row) + ' |')
    # edges/node
    epn_row = ['edges / node']
    for i in range(len(runs)):
        nc = cohort_aspects[i].get('node_count', 0)
        ec = cohort_aspects[i].get('edge_count', 0)
        epn_row.append(f'{(ec/nc) if nc else 0:.2f}')
    out.append('| ' + ' | '.join(epn_row) + ' |')
    out.append('')

    # ─── Per-item gold-anchor hits ─────────────────────────────────
    out.append('## Per-item: did the gold-required event get event_time?')
    out.append('')
    out.append('For each item, the row shows what the encoder wrote for the '
               'gold-required event (matched by title/content keywords). '
               'If the encoder did NOT write a dated node matching the gold '
               'event, the cell is empty.')
    out.append('')
    h = '| Item | ' + ' | '.join(labels) + ' | Expected |'
    out.append(h)
    out.append('|---|' + '---|' * (len(labels) + 1))
    for qid in sorted(items_by_qid.keys()):
        probe = ITEM_PROBES.get(qid, {})
        label = probe.get('label') or qid
        gold_keys = probe.get('gold_keywords') or []
        expected = probe.get('expected_event_time') or '—'
        row = [f'{qid} ({label})']
        for run in runs:
            b = items_by_qid[qid].get(run) or {}
            if not gold_keys:
                row.append('—')
                continue
            hit = _gold_anchor_hit(b, gold_keys)
            if hit:
                exp = probe.get('expected_event_time') or ''
                ok = '✓' if (not exp or hit['event_time'].startswith(exp)) else '~'
                row.append(f'{ok} {hit["event_time"]} ({hit["type"]})')
            else:
                row.append('✗ no anchor')
        row.append(expected)
        out.append('| ' + ' | '.join(row) + ' |')
    out.append('')

    # ─── Per-item full event_time inventory (for inspection) ───────
    out.append('## Per-item: full event_time inventory')
    out.append('')
    out.append('Every node with event_time set, sorted chronologically. '
               'This is the encoder\'s "temporal scaffolding" — what the '
               'downstream pipeline can reason about.')
    out.append('')
    for qid in sorted(items_by_qid.keys()):
        out.append(f'### `{qid}` — {ITEM_PROBES.get(qid, {}).get("label", "")}')
        out.append('')
        for run, label in zip(runs, labels):
            b = items_by_qid[qid].get(run) or {}
            dated = _dated_events(b)
            out.append(f'**{label}** ({len(dated)} dated nodes / {len((b.get("nodes") or []))} total):')
            if not dated:
                out.append('  _(none)_')
            else:
                for d in dated:
                    quotes = ''
                    if d['user_quote'] and d['anchor_quote']:
                        quotes = ' (u+a)'
                    elif d['user_quote']:
                        quotes = ' (u)'
                    elif d['anchor_quote']:
                        quotes = ' (a)'
                    out.append(f'  - `{d["event_time"]}` [{d["type"]}]{quotes} {d["title"]}')
            out.append('')

    return '\n'.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', help='run names to compare')
    p.add_argument('--out', default=None, help='markdown output path')
    p.add_argument('--labels', default=None,
                   help='comma-separated labels (defaults to short run names)')
    args = p.parse_args()

    runs = args.runs
    labels = args.labels.split(',') if args.labels else [
        # derive short labels from run names
        r.split('_')[1] if '_' in r else r for r in runs
    ]
    if len(labels) != len(runs):
        labels = runs

    # Collect bundles per qid per run (intersection of items)
    all_qids = set()
    by_run: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in runs:
        qids = set(list_items(run))
        by_run[run] = {}
        for qid in qids:
            try:
                by_run[run][qid] = load_artifacts(run, qid)
            except Exception as e:
                print(f'  load failed run={run} qid={qid}: {e}')
        all_qids = all_qids | qids
    common = sorted(q for q in all_qids if all(q in by_run[r] for r in runs))

    items_by_qid = {qid: {run: by_run[run][qid] for run in runs} for qid in common}

    print(f'comparing {len(common)} common items across {len(runs)} runs', flush=True)

    md = render_report(runs, items_by_qid, labels)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
        print(f'wrote {args.out}')
    else:
        print(md)


if __name__ == '__main__':
    main()
