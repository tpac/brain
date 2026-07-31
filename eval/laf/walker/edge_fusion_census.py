"""Edge-conditioned fusion census — is the CROSS-LANE edge pattern in the data?

Tom's fusion thesis (2026-07-29): the pairwise edge operation belongs INSIDE
the fusion, while lane identity is still alive — "Node A from lane cosine and
Node B from lane episodic are connected with an inhibitory edge; through the
fusion of lanes you apply different math based on their connection."

That predicts something checkable BEFORE any mechanism exists (the
role_readout law — ceiling before fitting): a node's goldness should depend on
WHICH lane lit its edge-neighbours, per verb class. If the (verb × neighbour-
lane) cell separates gold from non-gold AT THE SAME BASE STANDING, the
information is real and lane-conditioning earns its complexity. If the cells
separate identically regardless of neighbour lane, provenance is decoration
and post-fusion spread is enough.

POPULATION  corpus-v2 valids (quality era), per turn the ACTIVATED SET =
            top-200 by the shipped mix — the ordering-exam scope where a
            fusion term would act (reach is excitation's job; ordering is
            where inhibition has leverage at all).
FEATURE     per node n, per (verb_class, lane_group) cell: the max z of any
            edge-neighbour m that is LIT in that lane group (z ≥ LIT_Z),
            bidirectional (stored direction carries no traversal signal —
            c9d8f472), time-honest (edges created after the turn excluded).
LANE GROUPS current  = maxsim/sit/idf (dense cosine on the current message)
            episodic = pick/enc (past-moment roles)
            history  = M_h (the moment/history field)
VERB CLASS  corrective_strict (replacement verbs) / corrective_soft (the rest
            of the correction_improvement aspect — the hygiene verb-split
            claimed these are mostly complementary; this tests that) /
            complementary / similarity / hebbian / structural / temporal
READOUT     within base-rank bands (1–5 / 6–25 / 26–100 / 101–200), presence
            rate and magnitude for GOLD vs NON-GOLD. Gold-enriched cell →
            excitation licensed; gold-depleted → inhibition licensed. Band
            stratification is what makes it marginal information rather than
            a restatement of the base score.

Read-only. Run:  VECLIB_MAXIMUM_THREADS=3 nice -n 19 \
                 ./dev python3 eval/laf/walker/edge_fusion_census.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REPORT = OUT_DIR / 'edge_fusion_census.md'

LAM = 0.65           # production current/history blend
TOP_N = 200          # activated set per turn (ordering-exam scope)
LIT_Z = 1.0          # a lane "lit" a node if its z in that lane ≥ this
BANDS = ((1, 5), (6, 25), (26, 100), (101, TOP_N))
LANE_GROUPS = {'current': ('maxsim', 'sit', 'idf'),
               'episodic': ('pick', 'enc')}      # 'history' handled from M_h

CORRECTIVE_STRICT = {'corrects', 'corrected_by', 'supersedes', 'superseded_by',
                     'updates', 'overrides', 'revises', 'redefines',
                     'could_replace', 'preferred_over', 'rejected_for',
                     'restates', 'changes', 'modifies'}
SIMILARITY = {'similar_to', 'related', 'related_to', 'parallels',
              'same_domain_as', 'differs_from', 'parallels_design_of'}
HEBBIAN = {'co_accessed', 'co_anchored'}
STRUCTURAL = {'community_member'}
TEMPORAL = {'before', 'after', 'during', 'simultaneous_with', 'anchored_to'}


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def verb_class(rel, corrective_all):
    if rel in CORRECTIVE_STRICT:
        return 'corrective_strict'
    if rel in corrective_all:
        return 'corrective_soft'
    if rel in SIMILARITY:
        return 'similarity'
    if rel in HEBBIAN:
        return 'hebbian'
    if rel in STRUCTURAL:
        return 'structural'
    if rel in TEMPORAL:
        return 'temporal'
    return 'complementary'


def main():
    aspects = json.loads(
        (REPO / 'servers/scales/s2/aspects_v1.json').read_text())
    corrective_all = set(
        (aspects.get('correction_improvement') or {}).get('edge_relations') or [])
    if not corrective_all:
        raise SystemExit('correction_improvement aspect missing')

    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    n_nodes = idx['n_nodes']

    # bidirectional adjacency: node_i -> [(other_i, vclass, created_at)]
    b = open_brain_ro()
    adj = defaultdict(list)
    n_er = 0
    for src, tgt, rel, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        vc = verb_class(rel, corrective_all)
        adj[si].append((ti, vc, created))
        adj[ti].append((si, vc, created))
        n_er += 1
    b.close()
    print('adjacency: %d edge-relations over master, %d nodes with edges'
          % (n_er, len(adj)))

    cells = sorted({verb_class(r, corrective_all)
                    for r in (set(SIMILARITY) | HEBBIAN | STRUCTURAL | TEMPORAL
                              | CORRECTIVE_STRICT | corrective_all
                              | {'extends'})})
    groups = list(LANE_GROUPS) + ['history']
    # stats[(band, vclass, group)] = [gold_with, gold_tot, non_with, non_tot,
    #                                 gold_val_sum, non_val_sum]
    stats = defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0])
    by_door = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0.0, 0.0]))
    gains = np.array([A.GAINS[ln] for ln in A.LANES])
    n_turns = n_gold_in_set = 0

    for clabel, cutoff in (('quality (≥2026-05-11)', '2026-05-11'),):
        turns, _enr, n = D.build_corpus(cutoff)
        for t in turns:
            U = np.flatnonzero(t['alive'])
            if U.size < 20:
                continue
            Z = np.column_stack([t['zl'][ln][U] for ln in A.LANES])
            f0 = Z @ gains
            if not np.isfinite(f0).any() or f0.std() <= 1e-9:
                continue
            zf0 = (f0 - f0.mean()) / f0.std()
            zmh = zn(t['mh'])[U]
            mix = LAM * zf0 + (1.0 - LAM) * zmh    # production blend
            fin = np.where(np.isfinite(mix), mix, -np.inf)
            local_order = np.argsort(-fin)
            top_local = local_order[:TOP_N]
            top_nodes = U[top_local]
            gpos = np.flatnonzero(top_nodes == t['gr'])
            n_turns += 1
            if gpos.size == 0:
                continue                      # gold beyond the activated set
            n_gold_in_set += 1
            gold_rank = int(gpos[0]) + 1

            # per-node lane-group z, full-length arrays for O(1) lookup
            lit = {}
            for gname, lanes in LANE_GROUPS.items():
                arr = np.full(n_nodes, -np.inf, dtype=np.float64)
                stack = np.vstack([t['zl'][ln][U] for ln in lanes])
                arr[U] = np.nanmax(stack, axis=0)
                lit[gname] = arr
            harr = np.full(n_nodes, -np.inf, dtype=np.float64)
            harr[U] = zmh
            lit['history'] = harr

            turn_dt = t.get('turn_dt')
            door = 'door-1' if t['stratum'] == 'cue' else 'door-2'
            for pos, ni in enumerate(top_nodes, 1):
                band = next((lo, hi) for lo, hi in BANDS if lo <= pos <= hi)
                is_gold = (ni == t['gr'])
                feat = defaultdict(float)     # (vclass, group) -> max z
                for (oi, vc, created) in adj.get(int(ni), ()):
                    edt = iso(created)
                    if turn_dt and edt and edt > turn_dt:
                        continue
                    for gname in groups:
                        z = lit[gname][oi]
                        if z >= LIT_Z and z > feat[(vc, gname)]:
                            feat[(vc, gname)] = z
                for vc in cells:
                    for gname in groups:
                        v = feat.get((vc, gname), 0.0)
                        for tgt_stats in (stats[(band, vc, gname)],
                                          by_door[door][(band, vc, gname)]):
                            if is_gold:
                                tgt_stats[1] += 1
                                tgt_stats[4] += v
                                if v > 0:
                                    tgt_stats[0] += 1
                            else:
                                tgt_stats[3] += 1
                                tgt_stats[5] += v
                                if v > 0:
                                    tgt_stats[2] += 1

    L = ['# Edge-conditioned fusion census — cross-lane edge context vs goldness',
         '',
         'turns: %d · gold inside top-%d activated set: %d (%.0f%%) · LIT_Z=%.1f'
         % (n_turns, TOP_N, n_gold_in_set,
            100.0 * n_gold_in_set / max(n_turns, 1), LIT_Z), '',
         'Presence rate = share of nodes having ANY lit neighbour in that '
         '(verb class × neighbour lane) cell, gold vs non-gold, WITHIN a base-'
         'rank band. LIFT > 1 → gold-enriched (excitation licensed); < 1 → '
         'gold-depleted (inhibition licensed).', '']
    for lo, hi in BANDS:
        rows = []
        for vc in cells:
            for gname in groups:
                gw, gt, nw, nt, gv, nv = stats[((lo, hi), vc, gname)]
                if gt == 0 or nt == 0 or (gw + nw) == 0:
                    continue
                gr_ = gw / gt
                nr_ = nw / nt
                rows.append((abs(np.log((gr_ + 1e-9) / (nr_ + 1e-9))),
                             vc, gname, gw, gt, gr_, nr_,
                             (gr_ / nr_) if nr_ > 0 else float('inf'),
                             gv / gt, nv / nt))
        if not rows:
            continue
        rows.sort(reverse=True)
        L += ['## base-rank band %d–%d  (golds here: %d)'
              % (lo, hi, stats[((lo, hi), cells[0], groups[0])][1]), '',
              '| verb class | neighbour lane | gold n | gold has | non-gold has'
              ' | LIFT | gold mean z | non mean z |',
              '|---|---|---|---|---|---|---|---|']
        for _s, vc, gname, gw, gt, gr_, nr_, lift, gm, nm in rows[:14]:
            L.append('| %s | %s | %d/%d | %.1f%% | %.1f%% | **%.2f×** | %.2f | %.2f |'
                     % (vc, gname, gw, gt, 100 * gr_, 100 * nr_, lift, gm, nm))
        L.append('')
    # cross-lane test: for each verb class, does the lift DIFFER by lane group?
    L += ['## Cross-lane test — does neighbour-lane change the verdict?', '',
          'Same verb class, different neighbour lane. If the lifts track each '
          'other, provenance is decoration; if they diverge (especially in '
          'sign), lane-conditioned fusion carries information post-fusion '
          'spread cannot.', '',
          '| band | verb class | lift(current) | lift(episodic) | lift(history) |',
          '|---|---|---|---|---|']
    for lo, hi in BANDS:
        for vc in cells:
            lifts = []
            for gname in groups:
                gw, gt, nw, nt, _gv, _nv = stats[((lo, hi), vc, gname)]
                lifts.append((gw / gt) / (nw / nt)
                             if gt and nt and nw else None)
            if all(x is None for x in lifts):
                continue
            L.append('| %d–%d | %s | %s |' % (
                lo, hi, vc,
                ' | '.join('%.2f×' % x if x is not None else '—'
                           for x in lifts)))
    # door split on the most informative band
    L += ['', '## Door split (band 6–25, the promotable zone)', '',
          '| door | verb class | neighbour lane | gold has | non-gold has | LIFT |',
          '|---|---|---|---|---|---|']
    for door in ('door-1', 'door-2'):
        rows = []
        for vc in cells:
            for gname in groups:
                gw, gt, nw, nt, _gv, _nv = by_door[door][((6, 25), vc, gname)]
                if not gt or not nt or not nw:
                    continue
                gr_, nr_ = gw / gt, nw / nt
                rows.append((abs(np.log((gr_ + 1e-9) / (nr_ + 1e-9))),
                             vc, gname, gr_, nr_, gr_ / nr_))
        rows.sort(reverse=True)
        for _s, vc, gname, gr_, nr_, lift in rows[:6]:
            L.append('| %s | %s | %s | %.1f%% | %.1f%% | **%.2f×** |'
                     % (door, vc, gname, 100 * gr_, 100 * nr_, lift))
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
