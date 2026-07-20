"""Mesh forensics (Tom's two asks, 2026-07-20): stop looking at aggregates.

1. REVERSE-ENGINEER the uncaptured headroom: on turns where the 2-way
   oracle gains @5 over the static Moment, which the router did NOT
   capture — which field held the gold, at what rank it peaked there,
   and how their readouts differ from the captured ones.
2. DISAGGREGATE: winner-field shares + per-arm reach sliced by gold node
   TYPE, gold node AGE at query time, cue length quartile, question-ness,
   and the A1-wins subset (turns where Anchor's own last msg's field holds
   the gold) — any conditioning signal that reduces variance.

Runs entirely on the field cache + walker turn meta + a READ-ONLY (mode=ro,
dashboard precedent) peek at brain.db for node type/created_at.

Run: ./dev python3 eval/laf/walker/mesh_forensics.py   (pool60 via env)
"""
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from field_mesh_probe import gold_rank                              # noqa: E402
from mesh_fit_probe import (Turn, ROUTER_FEATS, FOLDS,              # noqa: E402
                            newton_logistic, router_prob, oracle_label)
from field_mesh_probe import wsum                                   # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
FIELD_NAMES = ('F0', 'F1(op-1)', 'A1(anchor-1)', 'M_h', 'M_full')
AGE_EDGES = (1, 7, 30)            # days → buckets <1d, 1-7d, 7-30d, >30d


def node_meta():
    """{node_id: (type, created_at)} — read-only against brain.db."""
    db_dir = os.environ.get('BRAIN_DB_DIR') or \
        str(Path.home() / 'AgentsContext' / 'brain')
    p = Path(db_dir) / 'brain.db'
    conn = sqlite3.connect('file:%s?mode=ro' % p, uri=True)
    out = {nid: (t or '?', c or '') for nid, t, c in conn.execute(
        "SELECT id, type, created_at FROM nodes")}
    conn.close()
    return out


def age_bucket(created, ts):
    if not created or not ts:
        return '?'
    days = max(0.0, (np.datetime64(ts[:19]) - np.datetime64(created[:19]))
               / np.timedelta64(1, 'D'))
    for e in AGE_EDGES:
        if days < e:
            return '<%dd' % e
    return '>%dd' % AGE_EDGES[-1]


def main():
    walker = open_walker()
    tfeat = {(s, e, sq): (l or 0, q or 0)
             for s, e, sq, l, q in walker.execute(
                 "SELECT session_id, epoch, seq, op_len, has_question "
                 "FROM turns WHERE labeled=1")}
    cands_of = defaultdict(list)
    for s, e, q, nid in walker.execute(
            "SELECT session_id, epoch, seq, node_id FROM candidates "
            "WHERE node_id IS NOT NULL ORDER BY rowid"):
        cands_of[(s, e, q)].append(nid)
    walker.close()
    nmeta = node_meta()

    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    raw = [t for t in idx['turns'] if not t.get('skipped')]
    turns, gold_nid, ts_of = [], {}, {}
    for t in raw:
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.ro is None:
            continue
        cands = cands_of.get(tt.key, [])
        if t['gold_i'] >= len(cands):
            continue
        turns.append(tt)
        gold_nid[tt.key] = cands[t['gold_i']]
        ts_of[tt.key] = t['ts']
    sess = sorted({t.sess for t in turns})
    fold_of = {s: i % FOLDS for i, s in enumerate(sess)}
    print('turns %d' % len(turns))

    # refit the router per fold (identical to mesh_fit_probe)
    rt = {}
    for f in range(FOLDS):
        X, y = [], []
        for t in turns:
            if fold_of[t.sess] == f:
                continue
            lab = oracle_label(t)
            if lab is None:
                continue
            X.append([t.ro[k] for k in ROUTER_FEATS])
            y.append(lab)
        X, y = np.array(X), np.array(y, dtype=float)
        m = np.isfinite(X).all(axis=1)
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        rt[f] = (newton_logistic((X[m] - mu) / sd, y[m]), mu, sd)

    # per-turn record
    recs = []
    for t in turns:
        w, mu, sd = rt[fold_of[t.sess]]
        x = np.array([[t.ro[k] for k in ROUTER_FEATS]])
        w0 = float(router_prob(w, (x - mu) / sd)[0]) \
            if np.isfinite(x).all() else None
        sr = wsum([(w0, t.fields[0]), (1 - w0, t.mh)]) \
            if w0 is not None and t.mh is not None else t.mfull
        fmap = dict(zip(FIELD_NAMES,
                        (t.fields[0], t.fields[1], t.fields[2], t.mh,
                         t.mfull)))
        rks = {k: gold_rank(f, t.gr) if f is not None else None
               for k, f in fmap.items()}
        rr = gold_rank(sr, t.gr) if sr is not None else None
        cand = {k: (v or 10 ** 9) for k, v in rks.items()}
        winner = min(cand, key=cand.get)
        typ, created = nmeta.get(gold_nid[t.key], ('?', ''))
        recs.append({
            'key': t.key, 'rks': rks, 'router_rk': rr, 'w0': w0,
            'ro': t.ro, 'winner': winner,
            'type': typ, 'age': age_bucket(created, ts_of[t.key]),
            'op_len': tfeat.get(t.key, (0, 0))[0],
            'has_q': bool(tfeat.get(t.key, (0, 0))[1]),
        })

    hit = lambda r: r is not None and r <= 5   # noqa: E731

    # ---- 1. reverse-engineering the uncaptured headroom
    orc_gain = [r for r in recs
                if hit(r['rks']['F0']) and not hit(r['rks']['M_full'])]
    cap = [r for r in orc_gain if hit(r['router_rk'])]
    unc = [r for r in orc_gain if not hit(r['router_rk'])]
    print('\n== 1. oracle-2way GAIN turns (F0 hits @5, M_full misses): %d '
          '· router captured %d · UNCAPTURED %d ==' % (len(orc_gain),
                                                       len(cap), len(unc)))
    for name, grp in (('captured', cap), ('uncaptured', unc)):
        if not grp:
            continue
        print('  %s: w0 mean %.2f · gold rank in F0 %.1f · in M_full %.0f'
              % (name, np.mean([r['w0'] for r in grp if r['w0']]),
                 np.mean([r['rks']['F0'] for r in grp]),
                 np.median([r['rks']['M_full'] or 9999 for r in grp])))
        for k in ROUTER_FEATS:
            print('      %-10s mean %.3f' % (k, np.nanmean(
                [r['ro'][k] for r in grp])))
    # the 5-way-only headroom (gold hides in F1/A1/M_h, both F0+M_full miss)
    deep = [r for r in recs if not hit(r['rks']['F0'])
            and not hit(r['rks']['M_full'])
            and any(hit(r['rks'][k]) for k in ('F1(op-1)', 'A1(anchor-1)',
                                               'M_h'))]
    print('  deep-headroom turns (only a history field hits): %d — winners:'
          ' %s' % (len(deep), dict(Counter(r['winner'] for r in deep))))

    # ---- 2. disaggregation
    def slice_table(title, keyfn, min_n=25):
        groups = defaultdict(list)
        for r in recs:
            groups[keyfn(r)].append(r)
        print('\n== 2. %s ==' % title)
        print('  slice              n    M_full@5  router@5  oracle2@5'
              '   win: F0/M_full/A1/F1/M_h')
        for k in sorted(groups, key=lambda k: -len(groups[k])):
            g = groups[k]
            if len(g) < min_n:
                continue
            wc = Counter(r['winner'] for r in g)
            o2 = np.mean([hit(r['rks']['F0']) or hit(r['rks']['M_full'])
                          for r in g])
            print('  %-16s %5d   %5.1f%%    %5.1f%%    %5.1f%%'
                  '    %2.0f/%2.0f/%2.0f/%2.0f/%2.0f'
                  % (str(k)[:16], len(g),
                     100 * np.mean([hit(r['rks']['M_full']) for r in g]),
                     100 * np.mean([hit(r['router_rk']) for r in g]),
                     100 * o2,
                     *(100 * wc.get(n, 0) / len(g) for n in FIELD_NAMES)))

    slice_table('gold node TYPE', lambda r: r['type'])
    slice_table('gold node AGE at query', lambda r: r['age'])
    lens = sorted(r['op_len'] for r in recs)
    qs = [lens[int(len(lens) * f)] for f in (0.25, 0.5, 0.75)]
    slice_table('cue length quartile',
                lambda r: 'Q%d' % (1 + sum(r['op_len'] > q for q in qs)))
    slice_table('cue has question?', lambda r: r['has_q'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
