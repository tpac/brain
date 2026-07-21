"""INFLUENCE OPS (Tom's parallel-mechanisms go, 2026-07-21): eight
attention/graph-inspired influence operators as arms in ONE A/B harness,
on the honest baseline (per-msg K, λ static, support-z substrate).

The mechanism target (miss_anatomy 398e08fb): specific evidence loses to
generic mass — golds live in sparse-lane spikes, displacers ride maxsim;
BURIED golds sit under within-theme generic crowds. Arms:

  divisive/subtractive (noise thinners; ÷ in activation = − in z-space):
    occ        s − w·P, P = node's mix-top-5 occupancy rate across turns
               (attention-sink correction / behavioral genericity)
    deg        P = log1p(graph degree), normalized (ACT-R fan effect)
    base       P = node's mean composed-F0 z across turns (key-norm
               regularization — genericity by measurement)
    corr       P = 1 for nodes with an INCOMING correction-aspect edge
               (typed-edge inhibition: suppress the superseded, the
               untested inverse of the falsified corrector-transfer)
    meanfield  F0/M_h get the corpus-mean field subtracted BEFORE z (the
               DC / sink-token filter)
  sharpening:
    peak K     every slot field floor-clipped at its top-K threshold —
               the tail ties, only peaks discriminate (Tom's Q2 peak-sum)
  temporal:
    fatigue    s − w·1[node in previous turn's mix top-5], per session
               (running fatigue v1 — surfaced-recently inhibition)
  lateral:
    mexhat     greedy diversity rerank of the mix top-50 — each pick
               suppresses its cosine neighbors (Mexican-hat / MMR)

Metrics per arm: reach@5/@25 (all + gold-strong tier), churn vs baseline,
and rescues by prior-miss depth (6-25 / 26-100 / >100) — BURIED rescues
are the arm's target class. Occupancy/means are computed on a first pass
over the same turns (self-referential but label-free; the turn's own
contribution is a ~1/2000 sliver).

Machinery: Turn/lambda_star/zn imported (the wsum rule). Pool60 needs
BRAIN_DB_DIR (degree/edges/vectors read the brain).

Run:    ./dev python3 eval/laf/walker/influence_ops.py
Pool60: BRAIN_DB_DIR=~/AgentsContext/eval-corpus/0a9baa/pooled \
        WALKER_OUT_DIR=~/AgentsContext/eval-corpus/0a9baa/walker ...
"""
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from field_mesh_probe import wsum                                    # noqa: E402
from mesh_fit_probe import Turn, GAMMA                               # noqa: E402
from lambda_probe import zn, lambda_star, GRID                       # noqa: E402

CACHE = OUT_DIR / 'field_cache.npy'
INDEX = OUT_DIR / 'field_cache_index.json'
ASPECTS = REPO / 'servers' / 'scales' / 's2' / 'aspects_v1.json'
WEIGHTS = (0.5, 1.0, 2.0)
PEAK_KS = (50, 200)
MEX_W = (0.5, 1.0)
FAT_W = (0.5, 1.0)
BUCKETS = ((6, 25), (26, 100), (101, 10 ** 9))


def brain_conn():
    db = os.environ.get('BRAIN_DB_DIR') or \
        str(Path.home() / 'AgentsContext' / 'brain')
    return sqlite3.connect('file:%s/brain.db?mode=ro' % db, uri=True)


def rank_of(s, gr):
    if not np.isfinite(s[gr]):
        return None
    return int((np.where(np.isfinite(s), s, -np.inf) > s[gr]).sum()) + 1


def peak_clip(f, k):
    if f is None or np.isnan(f).all():
        return f
    fin = f[np.isfinite(f)]
    if len(fin) <= k:
        return f
    thr = np.partition(fin, -k)[-k]
    return np.where(np.isfinite(f), np.maximum(f, thr), np.nan)


def main():
    idx = json.loads(INDEX.read_text())
    fields = np.load(CACHE, mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    n = idx['n_nodes']
    master = idx['master']
    row_of = {nid: i for i, nid in enumerate(master)}

    turns = []
    for t in idx['turns']:
        if t.get('skipped'):
            continue
        tt = Turn(t, fields, S)
        if tt.gr >= 0 and tt.ro is not None and tt.mh is not None \
                and not np.isnan(tt.fields[0]).all():
            turns.append(tt)
    # session order for the fatigue arm
    turns.sort(key=lambda tt: (tt.key[0], tt.key[1], tt.key[2]))

    # static λ frame
    per_l = {l: 0 for l in GRID}
    for tt in turns:
        for l, r in lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr).items():
            per_l[l] += int(r <= 5)
    lam_s = max(GRID, key=lambda l: per_l[l])
    print('turns %d · static λ=%.2f' % (len(turns), lam_s))

    # ── pass 1: baseline mix per turn + occupancy + mean fields ────────
    base = {}                        # key → (mix ndarray kept? too big)
    occ = np.zeros(n)
    sumF0 = np.zeros(n)
    cntF0 = np.zeros(n)
    sumMh = np.zeros(n)
    cntMh = np.zeros(n)
    base_rank = {}
    for tt in turns:
        f0z, mhz = zn(tt.fields[0]), zn(tt.mh)
        both = np.isfinite(f0z) & np.isfinite(mhz)
        mix = np.where(both, (1 - lam_s) * mhz + lam_s * f0z, -np.inf)
        r = rank_of(np.where(np.isfinite(mix), mix, np.nan), tt.gr) \
            if np.isfinite(mix[tt.gr]) else None
        base_rank[tt.key] = r
        top5 = np.argsort(-mix)[:5]
        occ[top5] += 1
        fin0 = np.isfinite(tt.fields[0])
        sumF0[fin0] += tt.fields[0][fin0]
        cntF0[fin0] += 1
        finh = np.isfinite(tt.mh)
        sumMh[finh] += tt.mh[finh]
        cntMh[finh] += 1
    P_occ = occ / max(1.0, occ.max())
    meanF0 = np.where(cntF0 > 20, sumF0 / np.maximum(cntF0, 1), 0.0)
    meanMh = np.where(cntMh > 20, sumMh / np.maximum(cntMh, 1), 0.0)
    mF0z = np.abs(meanF0)
    P_base = mF0z / max(1e-9, mF0z.max())

    # ── brain-side inputs: degree, corrected-set, vectors ───────────────
    conn = brain_conn()
    deg = Counter()
    corr_verbs = set(json.loads(ASPECTS.read_text()).get(
        'correction_improvement', {}).get('edge_relations') or [])
    # Passive/reversed verbs: the TARGET is the corrector/survivor and the
    # SOURCE is the superseded node (review HIGH 2026-07-21 — 4% of
    # correction edges; blanket add(tgt) penalized up-to-date correctors).
    PASSIVE = {'corrected_by', 'addressed_by', 'resolved_by',
               'consolidated_into', 'absorbed_into', 'rejected_for'}
    corrected = set()
    # degree from PHYSICAL edges (edge×relation rows inflate ~13%)
    for src, tgt in conn.execute(
            "SELECT e.source_id, e.target_id FROM edges e "
            "JOIN nodes ns ON ns.id = e.source_id "
            "JOIN nodes nt ON nt.id = e.target_id "
            "WHERE ns.archived=0 AND nt.archived=0"):
        deg[src] += 1
        deg[tgt] += 1
    for src, tgt, rel in conn.execute(
            "SELECT e.source_id, e.target_id, er.relation FROM edges e "
            "JOIN edge_relations er ON er.edge_id = e.edge_id "
            "JOIN nodes ns ON ns.id = e.source_id "
            "JOIN nodes nt ON nt.id = e.target_id "
            "WHERE ns.archived=0 AND nt.archived=0"):
        if rel in corr_verbs:
            corrected.add(src if rel in PASSIVE else tgt)
    P_deg = np.zeros(n)
    for nid, d in deg.items():
        r = row_of.get(nid)
        if r is not None:
            P_deg[r] = np.log1p(d)
    P_deg /= max(1e-9, P_deg.max())
    P_corr = np.zeros(n)
    for nid in corrected:
        r = row_of.get(nid)
        if r is not None:
            P_corr[r] = 1.0
    vecs = np.zeros((n, 0))
    rowsv, blobs = [], []
    # _primary vectors live in node_enrichments (node_embeddings is a dead
    # v23-migration shell with 0 rows — review BLOCKER 2026-07-21: the
    # first mexhat run silently reported baseline as a 'tested' arm)
    try:
        rows_e = list(conn.execute(
            "SELECT node_id, embedding FROM node_enrichments "
            "WHERE vector_type='_primary' AND embedding IS NOT NULL"))
    except sqlite3.OperationalError:
        rows_e = []                  # corpus schemas without enrichments
    for nid, blob in rows_e:
        r = row_of.get(nid)
        if r is not None and blob:
            rowsv.append(r)
            blobs.append(np.frombuffer(blob, dtype=np.float32))
    conn.close()
    if blobs:
        dim = len(blobs[0])
        vecs = np.zeros((n, dim), dtype=np.float32)
        for r, v in zip(rowsv, blobs):
            if len(v) == dim:
                nv = np.linalg.norm(v)
                if nv > 1e-9:
                    vecs[r] = v / nv
    print('brain inputs: degree rows %d · corrected nodes %d · '
          'vectors %d/%d' % (int((P_deg > 0).sum()), len(corrected),
                             len(rowsv), n))

    # ── arms ────────────────────────────────────────────────────────────
    def evaluate(score_fn, needs_seq=False):
        """score_fn(tt, f0z, mhz, mix, prev5) → transformed score vector.
        Caveats when quoting (review 2026-07-21): turns an arm cannot
        score are DROPPED (per-arm n printed — cross-arm percentages are
        not strictly same-denominator); fatigue's prev5 is the arm's OWN
        transformed top-5 of the previous LABELED turn in the session
        (crosses epochs; own-vs-base-mix semantics moves churn by ~2)."""
        r5 = r25 = s5 = ns = m = 0
        gained = lost = 0
        resc = {b: [0, 0] for b in BUCKETS}   # rescued / total in bucket
        prev5 = {}
        for tt in turns:
            f0z, mhz = zn(tt.fields[0]), zn(tt.mh)
            both = np.isfinite(f0z) & np.isfinite(mhz)
            mix = np.where(both, (1 - lam_s) * mhz + lam_s * f0z, -np.inf)
            s = score_fn(tt, f0z, mhz, mix,
                         prev5.get(tt.key[0]) if needs_seq else None)
            if needs_seq:
                prev5[tt.key[0]] = np.argsort(
                    -np.where(np.isfinite(s), s, -np.inf))[:5]
            rb = base_rank.get(tt.key)
            if rb is None:
                continue
            rk = rank_of(np.where(np.isfinite(s), s, np.nan), tt.gr) \
                if np.isfinite(s[tt.gr]) else None
            if rk is None:
                continue
            m += 1
            r5 += int(rk <= 5)
            r25 += int(rk <= 25)
            if tt.strong:
                ns += 1
                s5 += int(rk <= 5)
            gained += int(rk <= 5 < rb)
            lost += int(rb <= 5 < rk)
            for b in BUCKETS:
                if b[0] <= rb <= b[1]:
                    resc[b][1] += 1
                    resc[b][0] += int(rk <= 5)
        return {'n': m, 'r5': 100 * r5 / max(1, m),
                'r25': 100 * r25 / max(1, m),
                's5': 100 * s5 / max(1, ns), 'gain': gained, 'lost': lost,
                'resc': resc}

    def pen_arm(P, w):
        Pz = P / max(1e-9, P.max())

        def fn(tt, f0z, mhz, mix, prev):
            return np.where(np.isfinite(mix), mix - w * Pz, np.nan)
        return fn

    def meanfield_fn(tt, f0z, mhz, mix, prev):
        f0 = tt.fields[0] - meanF0
        mh = tt.mh - meanMh
        f0z2, mhz2 = zn(np.where(np.isfinite(tt.fields[0]), f0, np.nan)), \
            zn(np.where(np.isfinite(tt.mh), mh, np.nan))
        both = np.isfinite(f0z2) & np.isfinite(mhz2)
        return np.where(both, (1 - lam_s) * mhz2 + lam_s * f0z2, np.nan)

    def peak_fn(k):
        def fn(tt, f0z, mhz, mix, prev):
            f0 = peak_clip(tt.fields[0], k)
            _f0, f1, a1, f2, a2 = tt.fields
            parts = [(GAMMA, peak_clip(a1, k)),
                     (GAMMA ** 2, peak_clip(f1, k)),
                     (GAMMA ** 3, peak_clip(a2, k)),
                     (GAMMA ** 4, peak_clip(f2, k))]
            mh = wsum([(w, f) for w, f in parts if f is not None])
            if mh is None:
                return np.full(len(f0), np.nan)
            z0, zh = zn(f0), zn(mh)
            both = np.isfinite(z0) & np.isfinite(zh)
            return np.where(both, (1 - lam_s) * zh + lam_s * z0, np.nan)
        return fn

    def fatigue_fn(w, recue_k=0):
        """recue_k>0 = Tom's 'inhibited UNLESS re-cued' exception: a
        previous-top-5 node the current msg's own field re-excites (F0
        top-recue_k) is exempt from the penalty."""
        def fn(tt, f0z, mhz, mix, prev):
            s = np.where(np.isfinite(mix), mix, np.nan)
            if prev is not None:
                s = s.copy()
                pen_rows = prev
                if recue_k:
                    f0top = set(np.argsort(-np.where(
                        np.isfinite(f0z), f0z, -np.inf))[:recue_k].tolist())
                    pen_rows = [r for r in prev if r not in f0top]
                s[pen_rows] -= w
            return s
        return fn

    def mexhat_fn(w):
        def fn(tt, f0z, mhz, mix, prev):
            s = np.where(np.isfinite(mix), mix, np.nan)
            if not vecs.shape[1]:
                return s
            top = np.argsort(-np.where(np.isfinite(s), s, -np.inf))[:50]
            sub = s[top].copy()
            order = []
            pen = np.zeros(len(top))
            for _ in range(len(top)):
                i = int(np.nanargmax(sub - pen))
                if not np.isfinite(sub[i]):
                    break
                order.append(top[i])
                sub[i] = -np.inf
                sims = vecs[top] @ vecs[top[i]]
                pen = np.maximum(pen, w * np.maximum(sims, 0))
            out = s.copy()
            for pos, row in enumerate(order):
                out[row] = 1000.0 - pos          # rerank top-50 positions
            return out
        return fn

    arms = [('baseline', lambda tt, f0z, mhz, mix, prev:
             np.where(np.isfinite(mix), mix, np.nan), False)]
    for w in WEIGHTS:
        arms.append(('occ w=%.1f' % w, pen_arm(P_occ, w), False))
        arms.append(('base w=%.1f' % w, pen_arm(P_base, w), False))
    for w in WEIGHTS[:2]:
        arms.append(('deg w=%.1f' % w, pen_arm(P_deg, w), False))
        arms.append(('corr w=%.1f' % w, pen_arm(P_corr, w), False))
    arms.append(('meanfield', meanfield_fn, False))
    for k in PEAK_KS:
        arms.append(('peak K=%d' % k, peak_fn(k), False))
    for w in FAT_W:
        arms.append(('fatigue w=%.1f' % w, fatigue_fn(w), True))
    for w in (0.5, 1.0):
        for k in (10, 25):
            arms.append(('fat-recue w=%.1f K=%d' % (w, k),
                         fatigue_fn(w, recue_k=k), True))
    for w in MEX_W:
        arms.append(('mexhat w=%.1f' % w, mexhat_fn(w), False))

    print('\n%-14s %6s %7s %7s %7s   %-9s %s'
          % ('arm', 'n', 'r@5', 'r@25', 'strong5', 'Δ(g/l)',
             'rescues 6-25 / 26-100 / >100'))
    for name, fn, seq in arms:
        r = evaluate(fn, needs_seq=seq)
        rs = ' / '.join('%d of %d' % tuple(r['resc'][b]) for b in BUCKETS)
        print('%-14s %6d %6.1f%% %6.1f%% %6.1f%%   +%d/-%d   %s'
              % (name, r['n'], r['r5'], r['r25'], r['s5'],
                 r['gain'], r['lost'], rs))
    return 0


if __name__ == '__main__':
    sys.exit(main())
