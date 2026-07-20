"""Forensics round 2 (Tom's go, 2026-07-20) — three targeted reads over the
field cache + graph substrate:

S1 CORRECTOR-TRANSFER INCIDENCE — how often is the gold a CORRECTOR
   (source of a correction_improvement-aspect edge) whose corrected partner
   outranks it in the field? The direct test of Tom's score-transfer
   ingredient (29072ead) before building any lane.
S2 A1-ROUTE AUTOPSY — the turns where my anchor msg -1's field wins:
   was the gold surfaced at the previous turn? created near it? what cue
   length / gold age / gold type do these turns carry?
S3 COMMUNITY-COHERENCE READOUTS — modal-community share of F0/M_h top-25,
   community agreement between them; AUC vs the 2-way oracle side (Tom's
   membership-vs-moment hypothesis).

Alignment gate: the cache index stores sha256(master[:n]); the engine is
reloaded and the prefix hash MUST match before any node_id→row lookup.

Run: ./dev python3 eval/laf/walker/mesh_forensics2.py    (live corpus)
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from field_mesh_probe import gold_rank, wsum                        # noqa: E402
from mesh_fit_probe import Turn                                     # noqa: E402
from soft_usage import auc                                          # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
ASPECTS = REPO / 'servers' / 'scales' / 's2' / 'aspects_v1.json'


def rank_of(f, row):
    if f is None or row is None or not np.isfinite(f[row]):
        return None
    return int((np.where(np.isfinite(f), f, -np.inf) > f[row]).sum()) + 1


def main():
    walker = open_walker()
    cands_of = defaultdict(list)
    for s, e, q, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE node_id IS NOT NULL ORDER BY rowid"):
        cands_of[(s, e, q)].append(nid)
    picked_at = defaultdict(set)
    for s, e, q, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE outcome='selected' AND node_id IS NOT NULL"):
        picked_at[(s, e, q)].add(nid)
    lab_seqs = defaultdict(list)
    for s, e, q in walker.execute(
            "SELECT session_id, epoch, seq FROM turns WHERE labeled=1 "
            "ORDER BY seq"):
        lab_seqs[(s, e)].append(q)
    op_len = {(s, e, q): l or 0 for s, e, q, l in walker.execute(
        "SELECT session_id, epoch, seq, op_len FROM turns WHERE labeled=1")}
    walker.close()

    corr_verbs = set(json.loads(ASPECTS.read_text()).get(
        'correction_improvement', {}).get('edge_relations') or [])
    print('correction verbs: %d' % len(corr_verbs))

    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}

    master = idx.get('master')
    assert master, ('index has no frozen master list — rebuild the field '
                    'cache (builder now stores it)')
    got = hashlib.sha256(('|'.join(master)).encode()).hexdigest()[:16]
    assert got == idx['master_hash'], 'index master list corrupt'
    n_cache = idx['n_nodes']
    print('master alignment gate: OK (%d rows, frozen in index)' % n_cache)
    row_of = {nid: i for i, nid in enumerate(master)}

    # graph data via READ-ONLY brain.db (dashboard precedent) — note the
    # acceptable skew: edges/types read from today's brain, fields frozen
    # at build time
    import os
    import sqlite3
    db_dir = os.environ.get('BRAIN_DB_DIR') or \
        str(Path.home() / 'AgentsContext' / 'brain')
    conn = sqlite3.connect('file:%s/brain.db?mode=ro' % db_dir, uri=True)
    corrected_by_gold = defaultdict(set)   # corrector -> {corrected}
    comm_of = defaultdict(set)
    for src, tgt, rel in conn.execute(
            "SELECT e.source_id, e.target_id, er.relation FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "JOIN nodes ns ON ns.id = e.source_id "
            "JOIN nodes nt ON nt.id = e.target_id "
            "WHERE ns.archived=0 AND nt.archived=0"):
        if rel in corr_verbs:
            corrected_by_gold[src].add(tgt)
        if rel == 'community_member':
            comm_of[tgt].add(src)
            comm_of[src].add(tgt)
    ntype = dict(conn.execute(
        "SELECT id, type FROM nodes WHERE archived=0"))
    created = dict(conn.execute(
        "SELECT id, created_at FROM nodes WHERE archived=0"))
    conn.close()
    comm_ids = {nid for nid, t in ntype.items() if t == 'community'}
    member_comms = {nid: (cs & comm_ids) for nid, cs in comm_of.items()}
    print('corrector nodes: %d · community members: %d'
          % (len(corrected_by_gold),
             sum(1 for v in member_comms.values() if v)))

    turns = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.ro is None:
            continue
        cands = cands_of.get(tt.key, [])
        if t['gold_i'] >= len(cands):
            continue
        gnid = cands[t['gold_i']]
        if ntype.get(gnid) == 'community':
            continue                       # gold hygiene (865afae)
        turns.append((tt, gnid, t['ts']))
    print('turns %d (ex-community golds)\n' % len(turns))

    # ---------- S1: corrector-transfer incidence ----------
    n_gold_corrector = partner_outranks = rescue = 0
    miss_total = rescue_of_miss = 0
    for tt, gnid, ts in turns:
        f = tt.mfull
        rg = rank_of(f, row_of.get(gnid))
        if rg is None:
            continue
        gold_miss = rg > 5
        miss_total += int(gold_miss)
        partners = corrected_by_gold.get(gnid, ())
        if not partners:
            continue
        n_gold_corrector += 1
        pranks = [rank_of(f, row_of.get(p)) for p in partners]
        pranks = [r for r in pranks if r is not None]
        if pranks and min(pranks) < rg:
            partner_outranks += 1
            if gold_miss and min(pranks) <= 5:
                rescue += 1
                rescue_of_miss += 1
    print('== S1 corrector-transfer incidence (M_full field) ==')
    print('  gold IS a corrector (has correction-out edges): %d/%d turns'
          % (n_gold_corrector, len(turns)))
    print('  corrected partner OUTRANKS the gold: %d' % partner_outranks)
    print('  RESCUE potential (gold misses @5, partner in top-5): %d '
          '(= %.1f%% of all %d misses)'
          % (rescue, 100 * rescue_of_miss / max(1, miss_total), miss_total))

    # ---------- S2: A1-route autopsy ----------
    a1_wins, others = [], []
    for tt, gnid, ts in turns:
        row = row_of.get(gnid)
        r_f0 = rank_of(tt.fields[0], row)
        r_mf = rank_of(tt.mfull, row)
        r_a1 = rank_of(tt.fields[2], row)
        if r_a1 is None:
            continue
        rec = (tt, gnid, ts)
        if (r_a1 <= 5 and (r_f0 is None or r_f0 > 5)
                and (r_mf is None or r_mf > 5)):
            a1_wins.append(rec)
        else:
            others.append(rec)

    def stats(group):
        surf_prev = fresh = 0
        lens, ages = [], []
        types = Counter()
        for tt, gnid, ts in group:
            s, e, q = tt.key
            seqs = lab_seqs.get((s, e), [])
            prev = max((x for x in seqs if x < q), default=None)
            if prev is not None and gnid in picked_at.get((s, e, prev),
                                                          set()):
                surf_prev += 1
            c = created.get(gnid) or ''
            if c and ts:
                d = (np.datetime64(ts[:19]) - np.datetime64(c[:19])) \
                    / np.timedelta64(1, 'D')
                ages.append(float(d))
                fresh += int(d < 1)
            lens.append(op_len.get(tt.key, 0))
            types[ntype.get(gnid, '?')] += 1
        n = max(1, len(group))
        return {'n': len(group),
                'surfaced_prev_turn': 100 * surf_prev / n,
                'gold_fresh_<1d': 100 * fresh / n,
                'median_age_d': float(np.median(ages)) if ages else -1,
                'median_cue_len': float(np.median(lens)) if lens else -1,
                'top_types': types.most_common(4)}
    print('\n== S2 A1-route autopsy (A1 hits @5, F0 and M_full both miss) ==')
    for name, g in (('A1-wins', a1_wins), ('all other turns', others)):
        st = stats(g)
        print('  %-16s n=%-4d surfaced@prev %.0f%% · gold<1d %.0f%% · '
              'median age %.1fd · median cue len %.0f · types %s'
              % (name, st['n'], st['surfaced_prev_turn'],
                 st['gold_fresh_<1d'], st['median_age_d'],
                 st['median_cue_len'], st['top_types']))

    # ---------- S3: community-coherence readouts ----------
    labels, r_share0, r_shareM, r_agree, r_gold_in = [], [], [], [], []
    hits = []
    for tt, gnid, ts in turns:
        f0, mh, mf = tt.fields[0], tt.mh, tt.mfull
        if mh is None:
            continue

        def modal(f):
            fin = np.where(np.isfinite(f), f, -np.inf)
            top = np.argpartition(-fin, 25)[:25]
            cc = Counter()
            for r in top:
                nid = master[r] if r < n_cache else None
                for c in member_comms.get(nid, ()):
                    cc[c] += 1
            if not cc:
                return None, 0.0
            c, k = cc.most_common(1)[0]
            return c, k / 25.0
        c0, s0 = modal(f0)
        cM, sM = modal(mh)
        row = row_of.get(gnid)
        r_f0 = rank_of(f0, row)
        r_mf = rank_of(mf, row)
        if r_f0 is None or r_mf is None or r_f0 == r_mf:
            side = None
        else:
            side = int(r_f0 < r_mf)
        gold_comms = member_comms.get(gnid, set())
        labels.append(side)
        r_share0.append(s0)
        r_shareM.append(sM)
        r_agree.append(float(c0 is not None and c0 == cM))
        r_gold_in.append(float(bool(gold_comms and c0 in gold_comms)))
        hits.append(int(r_mf <= 5))
    lab = np.array([x for x in labels if x is not None], dtype=float)
    keep = [i for i, x in enumerate(labels) if x is not None]
    print('\n== S3 community-coherence readouts (n=%d side-labeled, base '
          '%.2f) ==' % (len(lab), lab.mean()))
    for name, vals in (('comm_share_F0top25', r_share0),
                       ('comm_share_Mhtop25', r_shareM),
                       ('modal_comm_agree', r_agree),
                       ('gold_in_F0_modal_comm', r_gold_in)):
        v = np.array([vals[i] for i in keep])
        a = auc(v[lab == 1], v[lab == 0]) if lab.std() > 0 else float('nan')
        h = np.array(hits)[keep] if name == 'modal_comm_agree' else None
        print('  %-22s AUC(side) %.3f · mean %.3f'
              % (name, a, float(np.mean(v))))
    # does community agreement predict the MOMENT being right at all?
    v = np.array(r_agree)
    h = np.array(hits)
    print('  modal_comm_agree vs M_full gold-hit: AUC %.3f (hit rate '
          'agree %.1f%% vs disagree %.1f%%)'
          % (auc(v[h == 1], v[h == 0]),
             100 * h[v == 1].mean(), 100 * h[v == 0].mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
