"""Can the enrichment lane's CEILING be lifted? (seeds K, communities, corridors)

WHY. A sparse lane's reach ceiling is P(gold ∈ its support) — enrichment's
solo reach is identical at @10 and @25 because only ~13 nodes are ever lit
(056bb6d0). No gain can fix that; only a bigger support can. So before any
more gain work, measure what each support-widening policy buys and costs.

CONFIGURATION-FREE BY DESIGN. Every number here is computed WITHOUT gains,
weights, or a mix — purely "is the gold in the lit set, and how big is that
set". Per 312021a2 that puts these results in the class that survives any
later gain change, unlike the marginal verdicts measured earlier.

POLICIES
  seeds-K        maxsim-top-K seeds, base-union filter (K = 5/10/20/40)
  +community     admit co-members of communities containing >=m seeds
                 (community node --community_member--> member; sizes are
                 small here, mode 3-7, so this is cheap fan-out)
  +corridor      same but only communities flagged community_is_corridor —
                 loose-from-birth bundles (dbf9146e), so a REACH instrument
                 (spans distant regions) rather than a precision one
  BOOST NOT FILTER (48fcb7c3 / 73a98824): 31% of active nodes have no
  membership, so community structure may only ADD candidates. Nothing here
  gates on membership.

Reported per policy: support size, ceiling P(gold in support), rescuable
misses@5, and the MARGINAL TRADE (extra golds per extra support-node) — a
policy that doubles the lit set to buy one gold is a bad trade regardless of
how much its ceiling rises.

Read-only. Run:  ./dev python3 eval/laf/walker/enrichment_widen.py
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR, WALKER_DB, open_ro, open_brain_ro

sys.path.append(str(OUT_DIR))   # DATA dir may be another tree: append so
                                 # THIS tree's code wins, while main-tree-only
                                 # helpers (lambda_probe, miss_anatomy) resolve
from lambda_probe import zn                                          # noqa: E402
from miss_anatomy import rank_in                                     # noqa: E402
import enrichment_lane as EL                                        # noqa: E402
import laf_lane_audit as A                                          # noqa: E402

SEED_KS = (5, 10, 20, 40)
REPORT = OUT_DIR / 'enrichment_widen.md'


def load_communities(b, m2i):
    """(node_row -> set(community_row), community_row -> set(member_rows),
    set(corridor community_rows), community_row -> internal_fraction).

    Direction verified: the community node is the SOURCE of community_member,
    the member is the TARGET. `cohesion` (internal_fraction) is returned so a
    community-admitted node can be scored by REAL data — how much of its
    community's story is internal (77b2617c) — instead of a hand-picked
    constant. Corridors are low-cohesion by definition (dbf9146e), so this
    correctly makes loose-bundle admissions weak rather than uniform."""
    of_node = defaultdict(set)
    members = defaultdict(set)
    for src, tgt in b.execute(
            "SELECT e.source_id, e.target_id FROM edges e "
            "JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE r.relation='community_member' "
            "AND (r.archived IS NULL OR r.archived=0)"):
        ci, mi = m2i.get(src), m2i.get(tgt)
        if ci is None or mi is None:
            continue
        of_node[mi].add(ci)
        members[ci].add(mi)
    corridor = set()
    for nid, val in b.execute(
            "SELECT node_id, value FROM node_metadata_kv "
            "WHERE key='community_is_corridor'"):
        if str(val).lower() == 'true' and nid in m2i:
            corridor.add(m2i[nid])
    cohesion = {}
    for nid, val in b.execute(
            "SELECT node_id, value FROM node_metadata_kv "
            "WHERE key='community_internal_fraction'"):
        if nid in m2i:
            try:
                cohesion[m2i[nid]] = float(val)
            except (TypeError, ValueError):
                pass
    return of_node, members, corridor, cohesion


def community_expand(seeds, of_node, members, min_seeds, restrict=None):
    """Communities touched by >=min_seeds seeds -> their member rows.
    restrict: only consider communities in this set (e.g. corridors)."""
    hits = Counter()
    for s in seeds:
        for c in of_node.get(s, ()):
            if restrict is None or c in restrict:
                hits[c] += 1
    out = set()
    for c, k in hits.items():
        if k >= min_seeds:
            out |= members[c]
    return out


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    n = idx['n_nodes']
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}
    w = open_ro(WALKER_DB)
    qvecs = EL.build_qvecs(w)
    w.close()
    b = open_brain_ro()
    node_meta = EL.build_node_meta(b, m2i)
    adj = EL.build_adjacency(b, m2i)
    of_node, members, corridor, _cohesion = load_communities(b, m2i)
    b.close()
    print('communities: %d with members · %d corridors · %d nodes with '
          'membership (%.0f%% of master)'
          % (len(members), len(corridor), len(of_node),
             100 * len(of_node) / n))

    # ── per-turn precompute: seeds at max K, mix rank, gold row ──────────
    turns = []
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        bd = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not bd or (bd['ts'] or '') < A.CUTOFF:
            continue
        gi = t.get('gold_i')
        if gi is None:
            continue
        gr = t['cand_rows'][gi]
        if gr < 0:
            continue
        F = fields[t['row']].astype(np.float32)
        f0 = F[S['op0']]
        if np.isnan(f0).all() or not np.isfinite(f0[gr]):
            continue
        mh = A.moment_history(F, S, n)
        mix = A.LAM * zn(f0) + (1 - A.LAM) * zn(mh)
        r = rank_in(mix, gr)
        if r is None:
            continue
        seeds_max, sz = EL.seed_rows(lanes_mm, t['row'], S, n, k=max(SEED_KS))
        turns.append({'key': tuple(t['key']), 'gr': gr, 'mix_rank': r,
                      'stratum': v['stratum'], 'seeds_max': seeds_max,
                      'sz': sz, 'turn_dt': EL.iso(bd['ts']),
                      'qv': qvecs.get(tuple(t['key']))})
    N = len(turns)
    miss5 = [t for t in turns if t['mix_rank'] > 5]
    print('turns %d · misses@5 %d\n' % (N, len(miss5)))

    L = ['# Enrichment support widening — can the ceiling move?', '',
         'n=%d clean valids ≥%s · misses@5 = %d · **configuration-free** '
         '(no gains, no mix — pure support membership)' % (N, A.CUTOFF, len(miss5)),
         '',
         'ceiling = P(gold ∈ lit set); rescuable@5 = misses whose gold is in '
         'the lit set (the reach a perfect scorer could convert).', '']

    def evaluate(policy_name, support_of):
        sizes, in_sup, resc = [], 0, 0
        for t in turns:
            sup = support_of(t)
            sizes.append(len(sup))
            if t['gr'] in sup:
                in_sup += 1
                if t['mix_rank'] > 5:
                    resc += 1
        return {'policy': policy_name, 'support': float(np.mean(sizes)),
                'ceiling': 100.0 * in_sup / N, 'rescuable': resc}

    def base_support(t, k):
        seeds = t['seeds_max'][:k]
        neigh = EL.aggregate_neighbors(seeds, t['sz'], adj, t['qv'],
                                       t['turn_dt'])
        return {oi for oi, d in neigh.items() if EL.passes_filter(d)}

    rows = []
    # A. seeds K sweep
    for k in SEED_KS:
        rows.append(evaluate('seeds K=%d (base union)' % k,
                             lambda t, _k=k: base_support(t, _k)))
        print('K=%d done' % k)

    # B. community expansion on top of K=5 and K=20
    for k in (5, 20):
        for m in (1, 2):
            rows.append(evaluate(
                'K=%d + community(≥%d seed%s)' % (k, m, '' if m == 1 else 's'),
                lambda t, _k=k, _m=m: base_support(t, _k) | community_expand(
                    t['seeds_max'][:_k], of_node, members, _m)))
            print('K=%d community m=%d done' % (k, m))

    # C. corridor-only expansion
    for k in (5, 20):
        rows.append(evaluate(
            'K=%d + corridor(≥1 seed)' % k,
            lambda t, _k=k: base_support(t, _k) | community_expand(
                t['seeds_max'][:_k], of_node, members, 1, restrict=corridor)))
        print('K=%d corridor done' % k)

    # D. community-only (no 1-hop) — isolates what communities alone buy
    for m in (1, 2):
        rows.append(evaluate(
            'community(≥%d) ONLY, no 1-hop' % m,
            lambda t, _m=m: community_expand(t['seeds_max'][:5], of_node,
                                             members, _m)))
    print('community-only done')

    base = rows[0]
    L += ['## Policies', '',
          'MARGINAL TRADE = extra rescuable golds per extra support-node/turn, '
          'measured against the K=5 base row. This is the column that decides: '
          'a policy that doubles the lit set to buy one gold is a bad trade no '
          'matter how much its ceiling rises.', '',
          '| policy | support (nodes/turn) | ceiling | rescuable@5 | '
          'Δsupport | Δrescuable | marginal trade |',
          '|---|---|---|---|---|---|---|']
    for r in rows:
        ds = r['support'] - base['support']
        dr = r['rescuable'] - base['rescuable']
        trade = ('—' if abs(ds) < 1e-9 else '%.2f golds / node' % (dr / ds))
        L.append('| %s | %.1f | %.0f%% | %d | %+.1f | %+d | %s |'
                 % (r['policy'], r['support'], r['ceiling'], r['rescuable'],
                    ds, dr, trade))
    L += ['', '- reference row: K=5 base union → support %.1f, ceiling %.0f%%, '
          'rescuable %d (the committed 52-gold figure).'
          % (base['support'], base['ceiling'], base['rescuable']), '']

    # ── which node classes does widening reach? (the abstract-type question)
    L += ['## Does widening reach the ABSTRACT types we systematically miss?',
          '', 'Gold type among rescuable misses, K=5 base vs the widest '
          'policy. The anatomy (ff93cce8) found rule/community/insight/lesson '
          'are the blind spot — if widening only adds concrete types it does '
          'not address it.', '',
          '| gold type | rescuable @ K=5 base | rescuable @ K=20+community(≥1) |',
          '|---|---|---|']
    def resc_types(support_of):
        c = Counter()
        for t in miss5:
            if t['gr'] in support_of(t):
                c[node_meta.get(t['gr'], (None, 0))[0]] += 1
        return c
    c_base = resc_types(lambda t: base_support(t, 5))
    c_wide = resc_types(lambda t: base_support(t, 20) | community_expand(
        t['seeds_max'][:20], of_node, members, 1))
    for ty in sorted(set(c_base) | set(c_wide),
                     key=lambda x: -(c_wide.get(x, 0))):
        L.append('| %s | %d | %d |' % (ty, c_base.get(ty, 0), c_wide.get(ty, 0)))
    L.append('')

    # ── stratum view
    L += ['## Rescuable@5 by stratum', '',
          '| stratum | misses | K=5 base | K=20+community(≥1) |',
          '|---|---|---|---|']
    for s in ('cue', 'window', 'session'):
        sub = [t for t in miss5 if t['stratum'] == s]
        nb = sum(1 for t in sub if t['gr'] in base_support(t, 5))
        nw = sum(1 for t in sub if t['gr'] in (
            base_support(t, 20) | community_expand(t['seeds_max'][:20],
                                                   of_node, members, 1)))
        L.append('| %s | %d | %d | %d |' % (s, len(sub), nb, nw))
    L.append('')

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
