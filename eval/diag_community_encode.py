#!/usr/bin/env python3
"""Diagnostic: dump the FULL decoder→encoder detail for v20 vs v21 so we can see
WHAT each arm actually proposed, decided, and said (journal) — not just counts.

Answers "why did v21 create fewer communities?" by showing: did the two arms
even get the same proposals (snapshot fidelity), and what did Haiku do/say.

    ./dev python3 eval/diag_community_encode.py [max_proposals] [batch_size]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.isolated_brain import IsolatedBrain
from eval.s2_community_decoder_eval import run_new_decoder
from servers.scales.s2.community_contract import COMMUNITY_DETECTION
from servers.scales.s2.community_encoder import CommunityEncoder
from eval.sim_community_structural import make_v21

MAX_PROPOSALS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def dump_arm(label, transform):
    print('\n' + '#' * 72)
    print('# ARM: %s' % label)
    print('#' * 72)
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        if transform is not None:
            v20 = brain.get_interaction_prompt('s2_community_enrichment') or ''
            v21 = transform(v20)
            params = brain.get_interaction_config('s2_community_enrichment') or {}
            reg = brain._interaction_dal.register(
                's2_community_enrichment', template=v21,
                parameters=json.dumps(params), created_by='diag:v21')
            brain._interaction_dal.set_active(
                's2_community_enrichment', reg['version'], set_by='diag:v21')

        dec = run_new_decoder(brain, dict(COMMUNITY_DETECTION))
        proposals = dec['proposals']
        by_type = {}
        for p in proposals:
            by_type[p['type']] = by_type.get(p['type'], 0) + 1
        print('\nDECODER proposals by type: %s' % by_type)

        news = [p for p in proposals if p['type'] == 'new_community']
        print('\nnew_community proposals (%d) — member_count · int_frac · '
              'first members:' % len(news))
        for p in news[:15]:
            mems = p.get('all_members', p.get('members', []))
            titles = ', '.join((m.get('title') or m.get('id', '?'))[:28]
                               for m in mems[:4])
            print('  mc=%-3s if=%.2f  %s%s' % (
                p.get('member_count', len(mems)),
                p.get('internal_fraction', 0),
                titles, ' …' if len(mems) > 4 else ''))

        cfg = dict(COMMUNITY_DETECTION)
        cfg['max_proposals_per_call'] = BATCH_SIZE
        cfg['max_actionable_per_run'] = MAX_PROPOSALS
        enc = CommunityEncoder(brain, config=cfg)
        result = enc.run(proposals, dec['community_state']) or {}

        print('\nENCODER result: rounds=%d actions=%d writes=%d' % (
            result.get('rounds', 0), result.get('actions', 0),
            result.get('write_actions', 0)))
        print('\naction_details:')
        for ad in (result.get('action_details') or []):
            print('  %s' % json.dumps(ad)[:300])

        print('\n--- final_text (Haiku reasoning + ## Review journal) ---')
        print(result.get('final_text', '') or '(empty)')

        # Persisted journal notes for this run.
        run_chain = enc.chain_id()
        rows = [r for r in brain.journal_notes(
                    scale='s2', unit='community_detection', k=1)
                if r.get('chain_id') == run_chain]
        print('\n--- persisted journal notes (%d) ---' % len(rows))
        for r in rows:
            print('  [%s] %s :: %s' % (
                r.get('tag') or '—', r.get('subject', ''),
                (r.get('note') or '')[:240]))


def main():
    dump_arm('v20 (writes structural fields)', None)
    dump_arm('v21 (structural fields dropped)', make_v21)


if __name__ == '__main__':
    main()
