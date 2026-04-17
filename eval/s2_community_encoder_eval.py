#!/usr/bin/env python3
"""S2 Community Encoder eval — rerun architecture against production agent.

Loads the live s2_community_enrichment interaction (prompt + model) from the
brain and runs the full decoder → rejection filter → encoder loop against an
IsolatedBrain copy. Fresh decode between cycles so proposals reflect the
updated graph after each encoder run.

Features:
1. Fingerprint suppression between cycles (via production rejection_table)
2. Priority-ordered proposals (merge > new_community > add_to_existing > health > drift)
3. Precise proposal→action matching; only genuinely skipped proposals get stamped
4. Optional --model override for A/B testing a different encoder model

Usage:
    python3 eval/s2_community_encoder_eval.py                    # 2 cycles
    python3 eval/s2_community_encoder_eval.py --cycles 3         # 3 cycles
    python3 eval/s2_community_encoder_eval.py --keep             # Keep temp dir
    python3 eval/s2_community_encoder_eval.py --save report.json
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse decoder harness from the decoder eval
from eval.s2_community_decoder_eval import (
    create_rejection_table,
    run_new_decoder,
    compute_cross_run_metrics,
)

# Production rejection + priority infrastructure — single source of truth
from servers.scales.s2.rejection_table import (
    match_proposals_to_actions,
    record_rejections,
    sort_proposals_by_priority,
)
from servers.scales.s2.community_contract import TYPE_PRIORITY




# ═══════════════════════════════════════════════════════════════
# ENCODER WRAPPER — runs the live S2 Community Encoder agent
# ═══════════════════════════════════════════════════════════════

def run_encoder(brain, proposals, community_state, max_proposals=20, model=None):
    """Run the S2 Community Encoder agent on priority-sorted proposals.

    The agent's prompt and default model live in the s2_community_enrichment
    interaction (loaded by the encoder from the brain). This function only
    overrides the model when --model is passed for A/B testing.

    Args:
        model: Optional model ID override (e.g. 'claude-haiku-4-5-20251001').
               If None, uses the live interaction's configured model.
    """
    from servers.scales.s2.community_encoder import CommunityEncoder
    from servers.scales.s2.community_contract import COMMUNITY_DETECTION

    config = dict(COMMUNITY_DETECTION)
    config['max_proposals_per_call'] = max_proposals
    config['max_actionable_per_run'] = max_proposals  # Single batch
    if model:
        config['model'] = model

    # If the caller passed a model override, write it into the interaction
    # config so _encode() picks it up. Prompt is unchanged — loaded from DB.
    if model:
        existing = brain.get_interaction_config('s2_community_enrichment') or {}
        existing['model'] = model
        brain._interaction_dal.register(
            's2_community_enrichment',
            template=brain.get_interaction_prompt('s2_community_enrichment') or '',
            parameters=json.dumps(existing),
            created_by='eval:model_override')

    encoder = CommunityEncoder(brain, config=config)

    # Sort by priority and cap
    sorted_proposals = sort_proposals_by_priority(proposals)[:max_proposals]

    model_name = model or config.get('model', 'default')
    print('\n  Encoder [%s] receiving %d proposals (of %d surviving):' % (
        model_name, len(sorted_proposals), len(proposals)))
    type_counts = Counter(p['type'] for p in sorted_proposals)
    for t, c in sorted(type_counts.items(), key=lambda x: TYPE_PRIORITY.get(x[0], 99)):
        print('    %-20s %d' % (t, c))

    t0 = time.time()
    result = encoder.run(sorted_proposals, community_state)
    elapsed = time.time() - t0

    if result:
        result['elapsed_s'] = round(elapsed, 1)
        result['proposals_sent'] = len(sorted_proposals)
        result['proposals_by_type'] = dict(type_counts)
        result['model'] = model_name
    else:
        result = {'elapsed_s': round(elapsed, 1), 'error': 'encoder failed',
                  'proposals_sent': len(sorted_proposals), 'model': model_name}

    return result, sorted_proposals


# ═══════════════════════════════════════════════════════════════
# FULL CYCLE: decode → filter → encode → record rejections
# ═══════════════════════════════════════════════════════════════

def run_cycle(brain, cycle_num, max_proposals=20, config=None, model=None):
    """One full decode→encode cycle.

    Returns dict with decoder stats, encoder stats, and actions taken.
    """
    model_label = model or 'default'
    print('\n' + '=' * 60)
    print('CYCLE %d [%s]' % (cycle_num, model_label))
    print('=' * 60)

    # Decode
    print('\n  Decoding...')
    decode_result = run_new_decoder(brain, config)
    proposals = decode_result['proposals']
    stats = decode_result['stats']

    print('  Unplaced: %d | Communities: %d' % (
        stats['unplaced'], stats['communities']))
    print('  Raw: %d | Suppressed: %d | Surviving: %d' % (
        stats['raw_proposals'], stats['suppressed_count'],
        stats['total_proposals']))
    if stats.get('suppressed_by_type'):
        print('  Suppressed: %s' % '  '.join(
            '%s=%d' % (t, c) for t, c in
            sorted(stats['suppressed_by_type'].items(), key=lambda x: -x[1])))

    if not proposals:
        print('  No proposals after suppression — converged.')
        return {
            'cycle': cycle_num,
            'decoder': stats,
            'encoder': None,
            'converged': True,
        }

    # Encode
    print('\n  Encoding with %s...' % (model or 'default model'))
    encode_result, sent_proposals = run_encoder(
        brain, proposals, decode_result['community_state'],
        max_proposals=max_proposals, model=model)

    actions = encode_result.get('write_actions', 0)
    rounds = encode_result.get('rounds', 0)
    elapsed = encode_result.get('elapsed_s', 0)

    print('\n  Encoder result: %d actions (%d writes) in %d rounds, %.1fs' % (
        encode_result.get('actions', 0), actions, rounds, elapsed))

    # Precise matching: determine which proposals the encoder actually acted on
    # by walking brain_batch operations from action_details. Only stamp the
    # skipped ones. Accepted proposals auto-invalidate on graph change anyway.
    total_sent = len(sent_proposals)
    acted_proposals, skipped_proposals = match_proposals_to_actions(
        sent_proposals, encode_result.get('action_details', []))

    # Detect encoder failure: API error, max_tokens explosion, or no rounds completed.
    # On failure, do NOT stamp — proposals deserve another chance next cycle.
    final_text = encode_result.get('final_text', '') or ''
    encoder_failed = (
        encode_result.get('rounds', 0) == 0 or
        bool(encode_result.get('error')) or
        'FAILED' in final_text or
        'ERROR' in final_text[:200]
    )

    if encoder_failed:
        print('  [!] Encoder failed or incomplete - NOT stamping proposals')
    else:
        print('  Matched: %d acted on, %d skipped (stamping skipped only)' % (
            len(acted_proposals), len(skipped_proposals)))
        if skipped_proposals:
            record_rejections(brain, skipped_proposals)

    return {
        'cycle': cycle_num,
        'decoder': stats,
        'encoder': {
            'proposals_sent': total_sent,
            'proposals_by_type': encode_result.get('proposals_by_type', {}),
            'actions': encode_result.get('actions', 0),
            'write_actions': actions,
            'rounds': rounds,
            'elapsed_s': elapsed,
            'final_text': encode_result.get('final_text', '')[:2000],
            'action_details': encode_result.get('action_details', []),
            'acted_on_count': len(acted_proposals),
            'skipped_count': len(skipped_proposals),
        },
        'converged': False,
    }


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def print_report(cycles, brain):
    """Print comprehensive eval report."""
    SEP = '=' * 70

    node_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE archived = 0").fetchone()[0]
    comm_count = brain.conn.execute(
        "SELECT COUNT(*) FROM nodes WHERE type = 'community' AND archived = 0"
    ).fetchone()[0]
    rejection_count = brain.conn.execute(
        "SELECT COUNT(*) FROM s2_rejections").fetchone()[0]

    print('\n' + SEP)
    print('S2 COMMUNITY ENCODER EVAL — SUMMARY')
    print(SEP)

    print('\nFinal state: %d nodes, %d communities' % (node_count, comm_count))
    print('Rejection table: %d entries' % rejection_count)

    # Trajectory
    print('\nPer-cycle trajectory:')
    print('  %-8s %-10s %-10s %-10s %-10s %-10s %s' % (
        'Cycle', 'Unplaced', 'Raw', 'Suppressed', 'Surviving', 'Actions', 'Time'))
    print('  ' + '-' * 70)
    for c in cycles:
        d = c['decoder']
        e = c.get('encoder') or {}
        print('  %-8d %-10d %-10d %-10d %-10d %-10d %s' % (
            c['cycle'],
            d['unplaced'],
            d['raw_proposals'],
            d['suppressed_count'],
            d['total_proposals'],
            e.get('write_actions', 0),
            '%.1fs' % e.get('elapsed_s', 0) if e else 'converged'))

    # Encoder journal excerpts
    for c in cycles:
        e = c.get('encoder')
        if e and e.get('final_text'):
            print('\n--- Cycle %d encoder journal ---' % c['cycle'])
            # Show first 1000 chars of journal
            journal = e['final_text'][:1000]
            for line in journal.split('\n'):
                print('  %s' % line)

    print('\n' + SEP)


def save_report(path, cycles):
    """Save report as JSON."""
    with open(path, 'w') as f:
        json.dump({
            'timestamp': datetime.utcnow().isoformat(),
            'cycles': cycles,
        }, f, indent=2, default=str)
    print('Saved to %s' % path)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='S2 Community Encoder Eval — rerun architecture with model comparison')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of decode→encode cycles (default: 2)')
    parser.add_argument('--max-proposals', type=int, default=20,
                        help='Max proposals per encoder call (default: 20)')
    parser.add_argument('--model', default=None,
                        help='Model override (e.g. claude-sonnet-4-6, claude-haiku-4-5-20251001)')
    parser.add_argument('--ab', action='store_true',
                        help='A/B test: cycle 1 with Sonnet 4.6, cycle 2 with Haiku 4.5')
    parser.add_argument('--keep', action='store_true',
                        help='Keep temp directory for inspection')
    parser.add_argument('--save', help='Save report to JSON file')
    args = parser.parse_args()

    from tests.isolated_brain import IsolatedBrain

    # Default A/B pair when --ab is used. Update when comparing new models.
    AB_MODELS = [
        'claude-sonnet-4-6',
        'claude-haiku-4-5-20251001',
    ]

    print('Setting up isolated brain copy...')
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        print('Isolated brain at: %s' % env.db_dir)

        # Create rejection table
        create_rejection_table(brain)

        cycles = []

        if args.ab:
            # A/B test: decode ONCE on the shared brain, then run each model
            # on a FRESH copy with the same proposals. This ensures both models
            # see identical input.
            print('\n*** A/B TEST: Sonnet 4.6 vs Haiku 4.5 ***')
            print('Decoding once, then each model gets identical proposals.\n')

            # Decode once to get proposals
            print('Decoding proposals...')
            decode_result = run_new_decoder(brain)
            all_proposals = decode_result['proposals']
            community_state = decode_result['community_state']
            stats = decode_result['stats']

            # Sort and cap to the proposals both models will see
            sorted_proposals = sort_proposals_by_priority(all_proposals)[:args.max_proposals]
            type_counts = Counter(p['type'] for p in sorted_proposals)

            print('  Unplaced: %d | Communities: %d' % (
                stats['unplaced'], stats['communities']))
            print('  Raw: %d | Surviving: %d | Sending: %d' % (
                stats['raw_proposals'], stats['total_proposals'],
                len(sorted_proposals)))
            for t, c in sorted(type_counts.items(),
                               key=lambda x: TYPE_PRIORITY.get(x[0], 99)):
                print('    %-20s %d' % (t, c))

            for model in AB_MODELS:
                # Fresh brain copy — encoder writes don't contaminate the other
                with IsolatedBrain(cleanup=not args.keep) as ab_env:
                    ab_brain = ab_env.brain
                    create_rejection_table(ab_brain)

                    print('\n' + '=' * 60)
                    print('MODEL: %s' % model)
                    print('=' * 60)
                    print('  Isolated at: %s' % ab_env.db_dir)

                    # Re-read community state from this copy (identical to original)
                    from servers.scales.s2.community_decoder import CommunityDecoder
                    ab_decoder = CommunityDecoder(ab_brain)
                    ab_community_state = ab_decoder._read_community_state()

                    print('  Encoding %d proposals with %s...' % (
                        len(sorted_proposals), model))

                    encode_result, sent = run_encoder(
                        ab_brain, all_proposals, ab_community_state,
                        max_proposals=args.max_proposals, model=model)

                    e = encode_result
                    print('\n  Result: %d actions (%d writes) in %d rounds, %.1fs' % (
                        e.get('actions', 0), e.get('write_actions', 0),
                        e.get('rounds', 0), e.get('elapsed_s', 0)))
                    print('  Tokens: %d in / %d out' % (
                        e.get('input_tokens', 0), e.get('output_tokens', 0)))

                    cycles.append({
                        'cycle': 1,
                        'model': model,
                        'decoder': stats,
                        'encoder': e,
                        'converged': False,
                    })

            # Print A/B comparison
            print('\n' + '=' * 70)
            print('A/B COMPARISON')
            print('=' * 70)
            print('\n  %-30s %-20s %-20s' % ('', AB_MODELS[0], AB_MODELS[1]))
            print('  ' + '-' * 70)

            def _get(idx, key, default=0):
                e = cycles[idx].get('encoder') or {}
                return e.get(key, default)

            print('  %-30s %-20s %-20s' % ('Actions (writes)',
                '%d (%d)' % (_get(0, 'actions'), _get(0, 'write_actions')),
                '%d (%d)' % (_get(1, 'actions'), _get(1, 'write_actions'))))
            print('  %-30s %-20s %-20s' % ('Rounds',
                _get(0, 'rounds'), _get(1, 'rounds')))
            print('  %-30s %-20s %-20s' % ('Time',
                '%.1fs' % _get(0, 'elapsed_s'),
                '%.1fs' % _get(1, 'elapsed_s')))
            print('  %-30s %-20s %-20s' % ('Input tokens',
                _get(0, 'input_tokens'), _get(1, 'input_tokens')))
            print('  %-30s %-20s %-20s' % ('Output tokens',
                _get(0, 'output_tokens'), _get(1, 'output_tokens')))
            print('  %-30s %-20s %-20s' % ('Truncations',
                len(_get(0, 'truncations', [])), len(_get(1, 'truncations', []))))

            for i, model in enumerate(AB_MODELS):
                e = cycles[i].get('encoder') or {}
                journal = e.get('final_text', '')
                if journal:
                    print('\n--- %s journal ---' % model)
                    for line in journal[:1500].split('\n'):
                        print('  %s' % line)
        else:
            for i in range(1, args.cycles + 1):
                result = run_cycle(
                    brain, i,
                    max_proposals=args.max_proposals,
                    model=args.model)
                cycles.append(result)

                if result.get('converged'):
                    print('\nConverged after %d cycles.' % i)
                    break

        print_report(cycles, brain)

        if args.save:
            save_report(args.save, cycles)

        if args.keep:
            print('\nTemp dir preserved: %s' % env.db_dir)


if __name__ == '__main__':
    main()
