#!/usr/bin/env python3
"""Run AspectIntegration N cycles against a cloned brain — production-equivalent code path.

Workflow:
  1. Clone production brain.db (via IsolatedBrain → temp dir).
  2. Copy aspects_v1.json to a temp file with member lists wiped — encoder
     classifies into description-only spec, no carry-over from production.
  3. Override ASPECTS_JSON_PATH + ASPECTS_PROPOSED_PATH env vars.
  4. Instantiate Brain → run AspectIntegration N times → dump final JSON.

The same code that runs in production runs here. Decoder, encoder, applier
all unchanged. Only the JSON paths and brain DB are swapped to temp copies.

Usage:
    ./dev python3 scripts/run_aspect_cycles_on_clone.py --cycles 15
    ./dev python3 scripts/run_aspect_cycles_on_clone.py --cycles 1 --keep-tempdir
    ./dev python3 scripts/run_aspect_cycles_on_clone.py --output /tmp/my_run.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument('--cycles', type=int, default=15,
                        help='How many cycles to run (default: 15 — covers ~373 strings at batch=30)')
    parser.add_argument('--output', default=None,
                        help='Output path for final aspects_v1.json (default: eval/aspects_v1_classified.json)')
    parser.add_argument('--keep-tempdir', action='store_true',
                        help="Don't clean up the temp dir — useful for poking at the audit trail")
    parser.add_argument('--seed-from', default=None,
                        help="Path to aspects_v1.json to start from (default: repo's seeded one)")
    parser.add_argument('--wipe-members', action='store_true',
                        help="Wipe pre-seeded member lists before running (harder eval — encoder re-derives "
                             "from descriptions alone, no anchors). Default: keep seeds as anchors (matches production).")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)

    # ── Set up the working aspects JSON ──────────────────────────────
    # Default: keep pre-seeded member lists as anchors — encoder pattern-matches
    # against known classifications, classifies only the long tail. This is
    # how production runs. With --wipe-members, encoder re-derives the whole
    # taxonomy (harder, less realistic).
    seed_path = args.seed_from or os.path.join(
        repo_root, 'servers', 'scales', 's2', 'aspects_v1.json')

    with open(seed_path, 'r') as f:
        spec = json.load(f)

    if args.wipe_members:
        starting_state = {}
        for name, aspect in spec.items():
            starting_state[name] = {
                'node_types': [],
                'edge_relations': [],
                'meaning': aspect.get('meaning', ''),
                'dimension': aspect.get('dimension', 'semantic'),
                'locked': aspect.get('locked', True),
                'metadata': aspect.get('metadata', {}),
            }
        print('starting from EMPTY member lists (--wipe-members) — harder eval, encoder re-derives taxonomy')
    else:
        starting_state = spec
        n_seed = sum(len(a.get('node_types', [])) + len(a.get('edge_relations', []))
                     for a in spec.values())
        print('starting from seeded aspects_v1.json (%d pre-classified anchors) — encoder classifies long tail only' % n_seed)

    work_dir = tempfile.mkdtemp(prefix='aspect_eval_')
    work_json = os.path.join(work_dir, 'aspects_v1.json')
    proposed_json = os.path.join(work_dir, 'aspects_proposed.json')

    with open(work_json, 'w') as f:
        json.dump(starting_state, f, indent=2)
        f.write('\n')

    os.environ['ASPECTS_JSON_PATH'] = work_json
    os.environ['ASPECTS_PROPOSED_PATH'] = proposed_json

    print('cloning brain into temp dir, working aspects JSON: %s' % work_json)

    # ── Run cycles inside IsolatedBrain ──────────────────────────────
    from tests.isolated_brain import IsolatedBrain
    from servers.scales.s2.aspect_integration import AspectIntegration

    summaries = []
    with IsolatedBrain(cleanup=False) as env:
        unit = AspectIntegration(env.brain, dispatch_fn=None)

        for i in range(1, args.cycles + 1):
            print('\n── cycle %d/%d ──' % (i, args.cycles))
            try:
                result = unit.run()
            except Exception as e:
                print('  FAILED: %s' % e)
                summaries.append({'cycle': i, 'error': str(e)})
                break

            if result.get('skipped'):
                print('  skipped: %s' % result['skipped'])
                summaries.append({'cycle': i, 'skipped': result['skipped'],
                                  'stats': result.get('stats', {})})
                if 'nothing unclassified' in result.get('skipped', ''):
                    print('  → all strings classified, stopping early.')
                    break
                continue

            print('  proposals: %d, classified: %d, rejected: %d, remaining: %d' % (
                result.get('proposals', 0),
                result.get('classified', 0),
                result.get('rejected', 0),
                result.get('remaining', 0)))
            per_aspect = result.get('per_aspect', {})
            if per_aspect:
                print('  per-aspect: %s' % ', '.join(
                    '%s=%d' % kv for kv in sorted(per_aspect.items(), key=lambda x: -x[1])))
            summaries.append({
                'cycle': i,
                'proposals': result.get('proposals', 0),
                'classified': result.get('classified', 0),
                'rejected': result.get('rejected', 0),
                'remaining': result.get('remaining', 0),
                'per_aspect': per_aspect,
                'stats': result.get('stats', {}),
            })

        if not args.keep_tempdir:
            print('\n(brain temp dir will be cleaned on exit — pass --keep-tempdir to inspect)')

    # ── Persist the final aspects JSON + run summary ─────────────────
    output = args.output
    if output is None:
        output = os.path.join(repo_root, 'eval', 'aspects_v1_classified.json')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    shutil.copy2(work_json, output)
    print('\nfinal aspects_v1.json: %s' % output)

    summary_path = output.replace('.json', '_run_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'cycles': summaries}, f, indent=2)
        f.write('\n')
    print('run summary:           %s' % summary_path)

    # Quick totals
    with open(output, 'r') as f:
        final = json.load(f)
    total_types = sum(len(a.get('node_types', [])) for a in final.values())
    total_relations = sum(len(a.get('edge_relations', [])) for a in final.values())
    print('\ntotal classified: %d node_types + %d edge_relations across %d aspects' % (
        total_types, total_relations, len(final)))

    if args.keep_tempdir:
        print('\ntemp dir kept at: %s' % work_dir)
        print('audit trail (last cycle): %s' % proposed_json)
    else:
        shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
