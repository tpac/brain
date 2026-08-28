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


# ═══════════════════════════════════════════════════════════════
# OPERATOR PROBES — which scoring operator should the gate use?
# ═══════════════════════════════════════════════════════════════

def build_matrices(emb, memberships):
    """Full node×centroid sim matrix for one space, own-community entries
    LOO-corrected (a member is scored as if it never contributed)."""
    ids = sorted(emb)
    idx = {n: i for i, n in enumerate(ids)}
    mat = np.stack([emb[n] for n in ids])
    cids = sorted(memberships)
    cent = np.stack([
        centroid_of(np.stack([emb[m] for m in memberships[c] if m in emb]))
        for c in cids])
    S = mat @ cent.T
    for j, c in enumerate(cids):
        mems = [m for m in memberships[c] if m in emb]
        if len(mems) < 2:
            continue
        s = np.sum([emb[m] for m in mems], axis=0)
        for m in mems:
            loo = unit((s - emb[m]) / (len(mems) - 1))
            S[idx[m], j] = float(np.dot(emb[m], loo))
    return ids, idx, mat, cids, S


def load_node_meta(brain):
    return {r[0]: (r[1], r[2] or '') for r in brain.conn.execute(
        "SELECT id, type, created_at FROM nodes WHERE archived = 0")}


def load_unplaceable_ids(brain):
    out = []
    for (blob,) in brain.conn.execute(
            "SELECT proposed_ids FROM s2_rejections "
            "WHERE proposal_type = 'unplaceable'"):
        try:
            out.extend(json.loads(blob))
        except (json.JSONDecodeError, TypeError):
            pass
    return set(out)


def load_recent_placements(brain, days=60):
    """(community_id, member_id) for community_member relations stamped by S2
    in the window — the accepted-placement pollution check."""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc)  # clock-ok — eval bookkeeping
              - timedelta(days=days)).isoformat()
    comm = {r[0] for r in brain.conn.execute(
        "SELECT id FROM nodes WHERE type = 'community'")}
    pairs = []
    for src, tgt, _ts in brain.conn.execute("""
        SELECT e.source_id, e.target_id, er.created_at
        FROM edges e JOIN edge_relations er ON er.edge_id = e.edge_id
        WHERE er.relation = 'community_member' AND er.archived = 0
        AND er.encoding_source LIKE 's2:%' AND er.created_at >= ?
    """, (cutoff,)):
        c, m = (src, tgt) if src in comm else (tgt, src)
        if c in comm and m not in comm:
            pairs.append((c, m))
    return pairs


def sims_for_pairs(S, idx, cid_pos, pairs):
    return np.array([S[idx[m], cid_pos[c]] for c, m in pairs
                     if m in idx and c in cid_pos])


def sample_null(S, nonmember_rows, n_pairs, rng):
    ni = rng.choice(nonmember_rows, size=n_pairs)
    ci = rng.integers(0, S.shape[1], n_pairs)
    return S[ni, ci]


def member_sim_array(S, idx, cids, memberships):
    vals, per_comm = [], {}
    for j, c in enumerate(cids):
        mem = [S[idx[m], j] for m in memberships[c] if m in idx]
        per_comm[c] = np.array(mem)
        vals.extend(mem)
    return np.array(vals), per_comm


def knn_vote_probe(ids, idx, mat, cids, S, memberships, member_of, rng,
                   k=10, nq=1000):
    """Choice quality: kNN-10 sim-weighted vote vs centroid argmax, on
    single-community members (clean truth)."""
    single = [m for m, cs in member_of.items() if len(cs) == 1 and m in idx]
    pick = rng.choice(len(single), size=min(nq, len(single)), replace=False)
    queries = [single[i] for i in pick]
    pool = sorted({m for c in cids for m in memberships[c] if m in idx})
    pos_of = {m: i for i, m in enumerate(pool)}
    sims = mat[[idx[m] for m in queries]] @ mat[[idx[m] for m in pool]].T
    knn_hit = cent_hit = 0
    for qi, m in enumerate(queries):
        row = sims[qi]
        if m in pos_of:
            row[pos_of[m]] = -np.inf
        votes = {}
        for t in np.argpartition(row, -k)[-k:]:
            for c in member_of[pool[t]]:
                votes[c] = votes.get(c, 0.0) + float(row[t])
        knn_hit += max(votes, key=votes.get) in member_of[m]
        cent_hit += cids[int(np.argmax(S[idx[m]]))] in member_of[m]
    n = len(queries)
    return knn_hit / n, cent_hit / n, n


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

        node_meta = load_node_meta(brain)
        unplaceable = load_unplaceable_ids(brain)
        placements = load_recent_placements(brain, days=60)

    print('\n' + '=' * 70)
    print('COMMUNITY PLACEMENT BASELINE — node vs centroid, production data')
    print('=' * 70)
    print_space(raw, [0.50, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85])
    print_space(cen, [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

    # ── operator probes (pure numpy from here on) ──
    ids, idx, _mat_r, cids, S_r = build_matrices(emb_raw, memberships)
    _, _, mat_c, cids_c, S_c = build_matrices(emb_c, memberships)
    assert cids == cids_c
    cid_pos = {c: j for j, c in enumerate(cids)}
    member_of = {}
    for c in cids:
        for m in memberships[c]:
            if m in idx:
                member_of.setdefault(m, set()).add(c)
    nonmember_rows = np.array(
        [i for i, n in enumerate(ids) if n not in member_of])

    print('\n' + '=' * 70)
    print('OPERATOR PROBES')
    print('=' * 70)

    # P1 — null stability: can the gate self-calibrate per run?
    p98s = np.array([
        float(np.percentile(sample_null(S_c, nonmember_rows, 2000, rng), 98))
        for _ in range(30)])
    print('\nP1 null stability (centred, 30 reps × 2000 pairs):')
    print('  p98 = %.4f ± %.4f  [min %.4f, max %.4f]' % (
        p98s.mean(), p98s.std(), p98s.min(), p98s.max()))

    # P2 — does centring change the CHOICE or only the cut?
    am_r, am_c = np.argmax(S_r, axis=1), np.argmax(S_c, axis=1)
    members = sorted(member_of)
    mrows = np.array([idx[m] for m in members])
    single = [m for m in members if len(member_of[m]) == 1]

    def _acc(am, who):
        return float(np.mean([cids[am[idx[m]]] in member_of[m] for m in who]))
    print('\nP2 choice (argmax community):')
    print('  argmax changed raw→centred: %.1f%% of all nodes, %.1f%% of members'
          % (100 * (am_r != am_c).mean(),
             100 * (am_r[mrows] != am_c[mrows]).mean()))
    print('  accuracy (argmax ∈ own communities) — members: raw %.3f → '
          'centred %.3f' % (_acc(am_r, members), _acc(am_c, members)))
    print('  accuracy, single-community members (n=%d): raw %.3f → '
          'centred %.3f' % (len(single), _acc(am_r, single), _acc(am_c, single)))

    # P3 — margin (top1 − top2, centred) as gate signal
    top2 = np.partition(S_c, -2, axis=1)
    margin = top2[:, -1] - top2[:, -2]
    m_mem, m_non = margin[mrows], margin[nonmember_rows]
    print('\nP3 margin top1−top2 (centred):')
    print('  members p25/p50/p75: %.4f / %.4f / %.4f' % tuple(
        np.percentile(m_mem, [25, 50, 75])))
    print('  nonmembers p25/p50/p75: %.4f / %.4f / %.4f' % tuple(
        np.percentile(m_non, [25, 50, 75])))
    for t in (0.02, 0.05, 0.10):
        print('  cut %.2f: member_keep %.3f, nonmember_pass %.3f' % (
            t, float((m_mem >= t).mean()), float((m_non >= t).mean())))

    # P4 — dispersion-z: "would this node look like a typical member?"
    _, per_comm = member_sim_array(S_c, idx, cids, memberships)
    mu = np.array([per_comm[c].mean() for c in cids])
    sd = np.array([per_comm[c].std() for c in cids])
    sd = np.maximum(sd, np.median(sd) * 0.5)  # small-n floor, disclosed
    Z = (S_c - mu[None, :]) / sd[None, :]
    z_mem = np.concatenate([(per_comm[c] - mu[j]) / sd[j]
                            for j, c in enumerate(cids)])
    z_non = sample_null(Z, nonmember_rows, 20000, rng)
    print('\nP4 dispersion-z (centred, per-community member mean/σ, '
          'σ floored at 0.5·median):')
    print('  member z mean %.2f σ %.2f; random z mean %.2f σ %.2f' % (
        z_mem.mean(), z_mem.std(), z_non.mean(), z_non.std()))
    for t in (-2.0, -1.5, -1.0, -0.5):
        print('  cut z≥%.1f: member_keep %.3f, random_pass %.3f' % (
            t, float((z_mem >= t).mean()), float((z_non >= t).mean())))

    # P5 — who is asleep: the unplaceable-rest census
    null_p98 = float(p98s.mean())
    unp = [m for m in unplaceable if m in node_meta]
    unp_rows = np.array([idx[m] for m in unplaceable if m in idx])
    from collections import Counter
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)  # clock-ok — eval bookkeeping
    buckets = Counter()
    for m in unp:
        ts = node_meta[m][1]
        try:
            days = (now - datetime.fromisoformat(ts)).days
        except ValueError:
            continue
        buckets['<30d' if days < 30 else '30–90d' if days < 90
                else '90–180d' if days < 180 else '>180d'] += 1
    print('\nP5 unplaceable rest census: %d marked, %d with embeddings' % (
        len(unplaceable), len(unp_rows)))
    print('  by type: %s' % ', '.join('%s=%d' % tc for tc in Counter(
        node_meta[m][0] for m in unp).most_common(8)))
    print('  by age: %s' % dict(buckets))
    if len(unp_rows):
        tops = S_c[unp_rows].max(axis=1)
        topz = Z[unp_rows].max(axis=1)
        print('  would place at centred≥p98(null)=%.3f: %d (%.1f%%)' % (
            null_p98, int((tops >= null_p98).sum()),
            100 * float((tops >= null_p98).mean())))
        print('  would place at z≥-1.0: %d (%.1f%%)' % (
            int((topz >= -1.0).sum()), 100 * float((topz >= -1.0).mean())))

    # P6 — recent accepted placements: the silent-pollution check
    pl = sims_for_pairs(S_c, idx, cid_pos, placements)
    print('\nP6 recent S2 placements (60d): %d scored' % len(pl))
    if len(pl):
        print('  centred sim p5/p25/p50/p75/p95: '
              '%.3f / %.3f / %.3f / %.3f / %.3f' % tuple(
                  np.percentile(pl, [5, 25, 50, 75, 95])))
        print('  below null p98 (%.3f): %.1f%%  — placements with no more '
              'geometric evidence than noise' % (
                  null_p98, 100 * float((pl < null_p98).mean())))

    # P7 — kNN-10 vote vs centroid argmax (choice quality, local operator)
    knn_acc, cent_acc, nq = knn_vote_probe(
        ids, idx, mat_c, cids, S_c, memberships, member_of, rng)
    print('\nP7 kNN-10 sim-weighted vote vs centroid argmax '
          '(%d single-community members):' % nq)
    print('  kNN vote accuracy %.3f vs centroid argmax %.3f' % (
        knn_acc, cent_acc))

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
