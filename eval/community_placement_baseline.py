#!/usr/bin/env python3
"""P0 baseline: node<->community-centroid cosine distributions (F13 gate).

The orphan placement gate (community_decoder._compute_orphan_affinities)
compares a node's `_primary` embedding against a community centroid — the
L2-normalized mean of member embeddings — with raw cosine against
`embedding_placement_threshold` (0.50). The measured node<->node random-pair
raw cosine is 0.6929 (anisotropy), but a centroid is an average and has its
own distribution. This measures the distributions the gate actually
discriminates between, on real production data:

  members   — member vs its OWN community centroid, leave-one-out (the
              decoder compares orphans, which never contribute to the
              centroid; LOO mirrors that geometry)
  other     — member of some community vs a DIFFERENT community's centroid
              (the hard negative: placing a node in the wrong community)
  random    — random non-member node vs random community centroid (the floor)

Each in two spaces: raw (what the gate uses today) and centred (subtract the
global mean embedding, renormalize — the emb_bench geometry.py definition),
plus a threshold sweep showing member-retention vs false-pass at each cut.

Read-only against an IsolatedBrain copy; never touches live data.

    ./dev python3 eval/community_placement_baseline.py
    ./dev python3 eval/community_placement_baseline.py --min-members 5 --save report.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEED = 20260827
RANDOM_PAIRS = 20000
OTHER_SAMPLE_PER_COMMUNITY = 30


def load_embeddings(brain):
    """{node_id: unit float32 vec} for live non-community nodes — the same
    population _compute_orphan_affinities scans."""
    vecs = {}
    for nid, blob in brain.conn.execute(
            "SELECT ne.node_id, ne.embedding FROM node_enrichments ne "
            "JOIN nodes n ON n.id = ne.node_id "
            "WHERE ne.vector_type = '_primary' AND ne.embedding IS NOT NULL "
            "AND n.archived = 0 AND n.type != 'community'"):
        vecs[nid] = np.frombuffer(blob, dtype=np.float32)
    return vecs


def load_memberships(brain):
    """{community_id: [member node_ids]} over live community_member edges,
    both edge directions, member side restricted to live non-community nodes."""
    members = defaultdict(list)
    for cid, mid in brain.conn.execute("""
        SELECT c.id,
               CASE WHEN e.source_id = c.id THEN e.target_id ELSE e.source_id END
        FROM nodes c
        JOIN edges e ON (e.source_id = c.id OR e.target_id = c.id)
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes m ON m.id = CASE WHEN e.source_id = c.id
                                    THEN e.target_id ELSE e.source_id END
        WHERE c.type = 'community' AND c.archived = 0
        AND er.relation = 'community_member' AND er.archived = 0
        AND m.archived = 0 AND m.type != 'community'
    """):
        members[cid].append(mid)
    return dict(members)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def centroid_of(mat):
    """Decoder-faithful centroid: L2-normalized mean of member vectors."""
    return unit(mat.mean(axis=0))


def measure_space(emb, memberships, rng, label):
    """One space's three distributions. `emb` is {id: unit vec} in that space."""
    member_sims = []
    centroids = {}
    member_mats = {}

    for cid, mids in memberships.items():
        mat = np.stack([emb[m] for m in mids if m in emb]) \
            if any(m in emb for m in mids) else None
        if mat is None or len(mat) < 2:
            continue
        member_mats[cid] = mat
        centroids[cid] = centroid_of(mat)
        # Leave-one-out: centroid without the member being scored.
        s = mat.sum(axis=0)
        for k in range(len(mat)):
            loo = unit((s - mat[k]) / (len(mat) - 1))
            member_sims.append(float(np.dot(mat[k], loo)))

    cids = sorted(centroids)
    all_ids = list(emb)
    member_of = {m for mids in memberships.values() for m in mids}

    other_sims = []
    for cid in cids:
        own = set(memberships[cid])
        pool = [m for m in member_of if m not in own and m in emb]
        if not pool:
            continue
        pick = rng.choice(len(pool), size=min(OTHER_SAMPLE_PER_COMMUNITY,
                                              len(pool)), replace=False)
        for k in pick:
            other_sims.append(float(np.dot(emb[pool[k]], centroids[cid])))

    non_members = [n for n in all_ids if n not in member_of]
    random_sims = []
    if non_members and cids:
        ni = rng.integers(0, len(non_members), RANDOM_PAIRS)
        ci = rng.integers(0, len(cids), RANDOM_PAIRS)
        for a, b in zip(ni, ci):
            random_sims.append(float(np.dot(emb[non_members[a]],
                                            centroids[cids[b]])))

    return {
        'label': label,
        'communities_measured': len(cids),
        'members': np.array(member_sims),
        'other': np.array(other_sims),
        'random': np.array(random_sims),
    }


def pct(a):
    if len(a) == 0:
        return {}
    q = np.percentile(a, [5, 25, 50, 75, 95])
    return {'n': len(a), 'mean': round(float(a.mean()), 4),
            'sigma': round(float(a.std()), 4),
            'p5': round(float(q[0]), 4), 'p25': round(float(q[1]), 4),
            'p50': round(float(q[2]), 4), 'p75': round(float(q[3]), 4),
            'p95': round(float(q[4]), 4)}


def sweep(space, thresholds):
    rows = []
    for t in thresholds:
        rows.append({
            'threshold': t,
            'member_keep': round(float((space['members'] >= t).mean()), 3),
            'other_pass': round(float((space['other'] >= t).mean()), 3),
            'random_pass': round(float((space['random'] >= t).mean()), 3),
        })
    return rows


def print_space(space, thresholds):
    print('\n--- %s ---' % space['label'])
    print('  communities measured: %d' % space['communities_measured'])
    for name in ('members', 'other', 'random'):
        p = pct(space[name])
        if not p:
            print('  %-8s (empty)' % name)
            continue
        print('  %-8s n=%-6d mean=%.4f σ=%.4f  '
              'p5=%.4f p25=%.4f p50=%.4f p75=%.4f p95=%.4f' % (
                  name, p['n'], p['mean'], p['sigma'],
                  p['p5'], p['p25'], p['p50'], p['p75'], p['p95']))
    print('  gap (member mean − random mean): %.4f' % (
        space['members'].mean() - space['random'].mean()))
    print('  %-9s %-12s %-11s %s' % ('cut', 'member_keep', 'other_pass',
                                     'random_pass'))
    for r in sweep(space, thresholds):
        print('  %-9.2f %-12.3f %-11.3f %.3f' % (
            r['threshold'], r['member_keep'], r['other_pass'],
            r['random_pass']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-members', type=int, default=5,
                    help='minimum members-with-embeddings per community')
    ap.add_argument('--save', help='write JSON report to this path')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain

    rng = np.random.default_rng(SEED)

    with IsolatedBrain() as env:
        brain = env.brain
        emb_raw = load_embeddings(brain)
        memberships = {
            cid: mids for cid, mids in load_memberships(brain).items()
            if sum(m in emb_raw for m in mids) >= args.min_members}

        print('nodes with _primary embedding: %d' % len(emb_raw))
        print('communities with >= %d embedded members: %d' % (
            args.min_members, len(memberships)))

        raw = measure_space(emb_raw, memberships, rng, 'RAW cosine')

        mean_vec = np.stack(list(emb_raw.values())).mean(axis=0)
        emb_c = {nid: unit(v - mean_vec) for nid, v in emb_raw.items()}
        cen = measure_space(emb_c, memberships, rng, 'CENTRED cosine')

    print('\n' + '=' * 70)
    print('COMMUNITY PLACEMENT BASELINE — node vs centroid, production data')
    print('=' * 70)
    print_space(raw, [0.50, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85])
    print_space(cen, [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    if args.save:
        payload = {}
        for space in (raw, cen):
            payload[space['label']] = {
                'communities_measured': space['communities_measured'],
                'distributions': {k: pct(space[k])
                                  for k in ('members', 'other', 'random')},
                'sweep': sweep(space,
                               [round(t, 2) for t in
                                np.arange(-0.1, 0.95, 0.05)]),
            }
        with open(args.save, 'w') as f:
            json.dump(payload, f, indent=2)
        print('\nSaved: %s' % args.save)


if __name__ == '__main__':
    main()
