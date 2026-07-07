"""Haiku selection bench — conversation-behavior scenarios, no encoding.

The frozen-corpus sweep gates end-to-end quality but its queries are
single-shot: the conversation window, per-turn <shown> dedup, and
thread-dependent picks are never exercised. This bench covers exactly that
layer at Haiku-only cost: fixed fixtures of (frame + turns + shown +
candidates + current message) with expected picks, rendered through the REAL
build_surface_prompt, one schema-enforced Haiku call each, scored against
gold. No brains, no encoding, no answerer, no judge.

Run:
    ./dev python3 eval/surface_turns_bench.py                 # both layouts, 3 reps
    ./dev python3 eval/surface_turns_bench.py --layout xml_v13 --reps 1
    ./dev python3 eval/surface_turns_bench.py --scenario shown_repick_trap

Scoring per rep: PASS iff every must_pick id is picked, no must_not_pick id
is picked, pick count <= max_picks, and (when present) exactly one of
pick_one_of. Hallucinated ids (outside the candidate set) fail the rep and
are counted separately.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.dispatch import load_env  # noqa: E402
load_env()

import anthropic  # noqa: E402
from servers.scales.s1.surface_contract import (  # noqa: E402
    SURFACE, SURFACE_MODEL, SURFACE_SELECTION_SCHEMA, build_surface_prompt,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'surface_turns_fixtures.json')
TEMPLATES = {
    'legacy': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'surface_v12_1_prompt.txt'),
    'xml_v13': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'surface_v13_prompt.txt'),
}


def _materialize_candidates(scenario):
    """Fixture candidates → render shape (created_at from days_ago)."""
    now = datetime.now(timezone.utc)
    out = []
    for c in scenario['candidates']:
        c = dict(c)
        days = c.pop('created_at_days_ago', 0)
        c['created_at'] = (now - timedelta(days=days)).isoformat()
        out.append(c)
    return out


def _shown_flat(scenario):
    """All surfaced entries across turns — the legacy layout's
    recently_recalled input (legacy sees shown ids as the separate
    'Recently surfaced' block; xml sees them per turn)."""
    flat = []
    for t in scenario.get('turns', []):
        flat.extend(t.get('surfaced') or [])
    return flat


def _build_prompt(scenario, layout):
    candidates = _materialize_candidates(scenario)
    kwargs = dict(
        recent_messages=scenario.get('turns', []),
        retrieval_stats=scenario.get('retrieval_stats'),
        frame=scenario.get('frame', ''),
        layout=layout,
    )
    if layout == 'legacy':
        kwargs['recently_recalled'] = _shown_flat(scenario)
    user_content, max_tokens = build_surface_prompt(
        candidates, scenario['current_message'], **kwargs)
    return user_content, max_tokens, {str(c['id'])[:8] for c in candidates}


def _call_haiku(client, system, user_content, max_tokens):
    resp = client.messages.create(
        model=SURFACE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        output_config={'format': {'type': 'json_schema',
                                  'schema': SURFACE_SELECTION_SCHEMA}},
    )
    return resp.content[0].text.strip()


def _score(scenario, picked_ids, candidate_ids):
    exp = scenario['expected']
    hallucinated = [p for p in picked_ids if p not in candidate_ids]
    missing = [m for m in exp.get('must_pick', []) if m not in picked_ids]
    forbidden = [m for m in exp.get('must_not_pick', []) if m in picked_ids]
    over_cap = len(picked_ids) > exp.get('max_picks', SURFACE['max_selected'])
    one_of = exp.get('pick_one_of')
    one_of_violation = False
    if one_of:
        n = len([p for p in picked_ids if p in one_of])
        one_of_violation = n != 1
    passed = not (hallucinated or missing or forbidden or over_cap
                  or one_of_violation)
    return {
        'pass': passed, 'picked': picked_ids, 'missing': missing,
        'forbidden_picked': forbidden, 'hallucinated': hallucinated,
        'over_cap': over_cap, 'one_of_violation': one_of_violation,
    }


def run_one(client, scenario, layout, system):
    user_content, max_tokens, candidate_ids = _build_prompt(scenario, layout)
    t0 = time.time()
    try:
        raw = _call_haiku(client, system, user_content, max_tokens)
        picked = [s.get('id', '') for s in json.loads(raw).get('selected', [])]
        result = _score(scenario, picked, candidate_ids)
    except Exception as e:
        result = {'pass': False, 'error': '%s: %s' % (type(e).__name__, e),
                  'picked': [], 'missing': [], 'forbidden_picked': [],
                  'hallucinated': [], 'over_cap': False,
                  'one_of_violation': False}
    result['latency_ms'] = int((time.time() - t0) * 1000)
    result['scenario'] = scenario['name']
    result['layout'] = layout
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', default='both',
                   choices=['both', 'legacy', 'xml_v13'])
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--scenario', default=None,
                   help='run a single scenario by name')
    p.add_argument('--template', default=None,
                   help='override the system template path for BOTH layouts')
    args = p.parse_args()

    with open(FIXTURES) as f:
        scenarios = json.load(f)['scenarios']
    if args.scenario:
        scenarios = [s for s in scenarios if s['name'] == args.scenario]
        if not scenarios:
            raise SystemExit('no scenario named %r' % args.scenario)

    layouts = ['legacy', 'xml_v13'] if args.layout == 'both' else [args.layout]
    systems = {}
    for lay in layouts:
        path = args.template or TEMPLATES[lay]
        with open(path) as f:
            systems[lay] = f.read()

    client = anthropic.Anthropic()
    jobs = [(s, lay) for lay in layouts for s in scenarios
            for _ in range(args.reps)]
    print('[bench] %d scenarios x %d layout(s) x %d rep(s) = %d Haiku calls'
          % (len(scenarios), len(layouts), args.reps, len(jobs)), flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda job: run_one(client, job[0], job[1], systems[job[1]]),
            jobs))

    # Per-layout, per-scenario table
    print()
    header = '%-32s' % 'scenario'
    for lay in layouts:
        header += ' %-10s' % lay
    print(header)
    print('-' * len(header))
    failures = []
    for s in scenarios:
        row = '%-32s' % s['name']
        for lay in layouts:
            reps = [r for r in results
                    if r['scenario'] == s['name'] and r['layout'] == lay]
            n_pass = sum(1 for r in reps if r['pass'])
            row += ' %d/%d       ' % (n_pass, len(reps))
            failures.extend(r for r in reps if not r['pass'])
        print(row)
    print()
    for lay in layouts:
        reps = [r for r in results if r['layout'] == lay]
        n_pass = sum(1 for r in reps if r['pass'])
        shown_viol = sum(1 for r in reps if r.get('forbidden_picked'))
        halluc = sum(1 for r in reps if r.get('hallucinated'))
        print('[%s] pass %d/%d  forbidden-pick reps: %d  hallucination reps: %d'
              % (lay, n_pass, len(reps), shown_viol, halluc))

    if failures:
        print('\nFailures detail:')
        for r in failures:
            detail = r.get('error') or (
                'picked=%s missing=%s forbidden=%s halluc=%s over_cap=%s one_of=%s'
                % (r['picked'], r['missing'], r['forbidden_picked'],
                   r['hallucinated'], r['over_cap'], r['one_of_violation']))
            print('  [%s/%s] %s' % (r['layout'], r['scenario'], detail))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'surface_turns_%s.json'
                            % datetime.now().strftime('%Y%m%d_%H%M%S'))
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'reps': args.reps,
                   'layouts': layouts}, f, indent=1)
    print('\n[bench] report → %s' % out_path)


if __name__ == '__main__':
    main()
