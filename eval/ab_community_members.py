#!/usr/bin/env python3
"""A/B: does removing `community_members` from the prompt make Haiku reliably
write the `connect_to` member edges?

Production measurement showed ~64% of newly-created communities are born with
ZERO member edges (Haiku writes the community_members string but omits the
edges); reconcile back-fills them from the declared list. Hypothesis: the
redundancy (members written twice — connect_to AND community_members) causes the
omission. Remove the easy string and the edges get written.

Arms (each on a fresh IsolatedBrain copy, real Haiku):
  v21 = live prompt (community_members present) — baseline.
  v22 = make_v22(v21) — community_members dropped, connect_to unchanged.

Metric (band-aid can't mask it): for each NEW community, count its member edges
by encoding_source. Agent edges = NOT 's2:community_repair'; reconcile edges =
's2:community_repair'. A community with ZERO agent edges = the agent omitted the
connect_to entirely. Report the omission rate per arm.

    ./dev python3 eval/ab_community_members.py [max_proposals] [batch_size]
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

MAX_PROPOSALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 6
REPAIR_SOURCE = 's2:community_repair'


def make_v22(v21: str) -> str:
    """Drop `community_members` from the NEW-COMMUNITY example AND the metadata
    doc-list. Each anchor must appear exactly once. connect_to is untouched."""
    edits = [
        ('\n   community_members: "577119fd: Hook pipeline latency, '
         '0fce53be: 20s root cause, 854b4bc3: Gemma resolution",', ''),
        ('\n- `community_members` — ALL member IDs as "id: title" pairs', ''),
    ]
    out = v21
    for old, new in edits:
        n = out.count(old)
        assert n == 1, 'v22 anchor not unique (count=%d): %r' % (n, old)
        out = out.replace(old, new)
    assert 'community_members' not in out, 'community_members still present'
    return out


def _edges_by_source(brain, cid):
    """{encoding_source: distinct_member_count} for a community's member edges."""
    rows = brain.conn.execute("""
        SELECT er.encoding_source, COUNT(DISTINCT member.id)
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes member ON member.id = CASE
            WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            AND member.archived = 0 AND member.type != 'community'
        WHERE er.relation = 'community_member' AND er.archived = 0
          AND (e.source_id = ? OR e.target_id = ?)
        GROUP BY er.encoding_source
    """, (cid, cid, cid)).fetchall()
    return {r[0]: r[1] for r in rows}


def run_arm(label, transform, max_prop, batch):
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        if transform is not None:
            v21 = brain.get_interaction_prompt('s2_community_enrichment') or ''
            v22 = transform(v21)
            params = brain.get_interaction_config('s2_community_enrichment') or {}
            reg = brain._interaction_dal.register(
                's2_community_enrichment', template=v22,
                parameters=json.dumps(params), created_by='ab:v22')
            brain._interaction_dal.set_active(
                's2_community_enrichment', reg['version'], set_by='ab:v22')

        pre = {r[0] for r in brain.conn.execute(
            "SELECT id FROM nodes WHERE type='community' AND archived=0").fetchall()}
        dec = run_new_decoder(brain, dict(COMMUNITY_DETECTION))
        proposals = dec['proposals']
        new_props = [p for p in proposals if p['type'] == 'new_community']
        if not new_props:
            print('  [%s] no new_community proposals' % label)
            return None

        cfg = dict(COMMUNITY_DETECTION)
        cfg['max_proposals_per_call'] = batch
        cfg['max_actionable_per_run'] = max_prop
        CommunityEncoder(brain, None, cfg).run(proposals, dec['community_state'])

        post = {r[0] for r in brain.conn.execute(
            "SELECT id FROM nodes WHERE type='community' AND archived=0").fetchall()}
        new_ids = post - pre

        created = omitted = agent_total = repair_total = 0
        detail = []
        for cid in new_ids:
            by_src = _edges_by_source(brain, cid)
            agent = sum(v for s, v in by_src.items() if s != REPAIR_SOURCE)
            repair = by_src.get(REPAIR_SOURCE, 0)
            created += 1
            agent_total += agent
            repair_total += repair
            if agent == 0:
                omitted += 1
            detail.append((cid[:8], agent, repair))
        return {
            'label': label, 'created': created, 'omitted': omitted,
            'agent_edges': agent_total, 'repair_edges': repair_total,
            'detail': sorted(detail, key=lambda x: x[1]),
        }


def main():
    print('=' * 64)
    print('A/B: community_members removal — agent edge-write completeness')
    print('=' * 64)
    a = run_arm('v21 (members present)', None, MAX_PROPOSALS, BATCH_SIZE)
    b = run_arm('v22 (members dropped)', make_v22, MAX_PROPOSALS, BATCH_SIZE)
    if not a or not b:
        print('inconclusive — an arm had no new_community proposals')
        return

    print('\n%-24s %9s %9s' % ('metric', 'v21', 'v22'))
    print('-' * 44)
    for name, k in [('communities created', 'created'),
                    ('  zero-agent-edge (omitted)', 'omitted'),
                    ('agent edges total', 'agent_edges'),
                    ('reconcile-repair edges', 'repair_edges')]:
        print('%-24s %9s %9s' % (name, a[k], b[k]))
    ra = a['omitted'] / a['created'] if a['created'] else 0
    rb = b['omitted'] / b['created'] if b['created'] else 0
    print('%-24s %8.0f%% %8.0f%%' % ('OMISSION RATE', ra * 100, rb * 100))

    for arm in (a, b):
        print('\n[%s] per-community (agent_edges, repair_edges):' % arm['label'])
        for cid, ag, rp in arm['detail']:
            print('   %s  agent=%-3d repair=%-3d%s' % (
                cid, ag, rp, '   <-- OMITTED' if ag == 0 else ''))


if __name__ == '__main__':
    main()
