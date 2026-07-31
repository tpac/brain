"""Anisotropy diagnostic — is the flat space fixable by whitening? LABEL-FREE.

The embedder maps all real text into a narrow cone: every query→node cosine
lands ~0.50–0.59 with sigma ~0.04, so all discrimination lives in ~+-0.08 of
range. The standard treatment for that pathology is to remove the distribution
mean and the top principal component(s) ("all-but-the-top" whitening).

DESIGNED TO BE JUDGED WITHOUT GOLD LABELS. The primary readouts are properties
of the SPACE, not of the corpus — so a verdict here does not depend on trusting
corpus-v2. Gold-rank effects are computed too, but reported separately and
clearly marked as the label-dependent half.

Arms (transform applied consistently to BOTH sides; each side centred by its
OWN mean, because the doc/query prefixes put the two distributions in different
places):
  raw · centre · centre+PC1 · centre+PC2 · centre+PC4 · centre+PC8

LABEL-FREE readouts
  1 ANISOTROPY   mean cosine between random node pairs (isotropic ~= 0),
                 ||mean vector|| / mean ||v||, PC1 variance share, and the
                 participation ratio (effective dimensionality).
  2 WHAT IS PC1  correlation of each node's PC1 projection with content length
                 and its distribution by node type — tells us whether the
                 dominant direction encodes something mundane (length, register)
                 that SHOULD be removed.
  3 DYNAMIC RANGE  query→node cosine mean/sigma per arm, and the head spread
                 (cos@1 - cos@25) per query. Bigger spread = more separable.
  4 CHURN        top-25 overlap vs raw — how violently the transform reorders.
LABEL-DEPENDENT (secondary)
  5 gold median rank / @5 / @25 per arm.

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 OMP_NUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/anisotropy_diagnostic.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker, open_brain_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from servers.recall_laf import LafV1Engine                          # noqa: E402

REPORT = OUT_DIR / 'anisotropy_diagnostic.md'
VIEW = '_primary'          # dominant view: supplies 61.5% of gold maxes
ARMS = [('raw', -1), ('centre', 0), ('centre+PC1', 1), ('centre+PC2', 2),
        ('centre+PC4', 4), ('centre+PC8', 8)]
TOPK = 25


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def tie_fair(scores, gi):
    gv = scores[gi]
    if not np.isfinite(gv):
        return None
    fin = np.where(np.isfinite(scores), scores, -np.inf)
    return int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']

    w = open_walker()
    qv_raw, keys, ts_of = [], [], {}
    for sess, epoch, seq, q, ts in w.execute(
            'SELECT session_id, epoch, seq, q_vec, ts FROM turns'):
        ts_of[(sess, epoch, seq)] = ts
        if q:
            qv_raw.append(np.frombuffer(q, dtype=np.float32).astype(np.float64))
            keys.append((sess, epoch, seq))
    w.close()
    qmap = {k: i for i, k in enumerate(keys)}
    Q0 = np.vstack(qv_raw)

    turns, _e, _n = D.build_corpus('2026-05-11')

    from tests.isolated_brain import IsolatedBrain
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
        n = eng._n
        M0 = np.asarray(eng._mats[VIEW][:n], dtype=np.float64)
        have = np.isfinite(M0).all(axis=1)
        M0 = M0[have]
        # Row-aligned creation stamps: the label-dependent half MUST mask nodes
        # that did not exist at turn time, or the rank denominator includes the
        # future. Captured here because the engine closes with this block.
        created_all = np.asarray(eng._created[:n], dtype='<U40')
        created = created_all[have]
        row_ids = [master_id for i, master_id in
                   enumerate(eng._master[:n]) if have[i]]
        rid = {sid: i for i, sid in enumerate(row_ids)}
        print('%s matrix: %d nodes x %d dims (queries %d)'
              % (VIEW, M0.shape[0], M0.shape[1], Q0.shape[0]))

    b = open_brain_ro()
    clen = {}
    typ = {}
    for nid, t_, L in b.execute(
            'SELECT id, type, LENGTH(content) FROM nodes'):
        clen[nid] = L or 0
        typ[nid] = t_
    b.close()

    # ── PCA on the DOCUMENT side (the large distribution) ──
    mu_d = M0.mean(axis=0)
    Xc = M0 - mu_d
    # economy SVD on 8k x 768 is cheap
    _U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    var_share = var / var.sum()
    part_ratio = (var.sum() ** 2) / (var ** 2).sum()
    mu_q = Q0.mean(axis=0)

    # PC1 semantics: projection vs content length and by type
    proj1 = Xc @ Vt[0]
    lens = np.array([clen.get(s, 0) for s in row_ids], dtype=float)
    ok = lens > 0
    r_len = float(np.corrcoef(proj1[ok], np.log1p(lens[ok]))[0, 1])

    L = ['# Anisotropy diagnostic — can whitening un-flatten the space?', '',
         'View `%s` · %d nodes · %d queries · **label-free readouts first**'
         % (VIEW, M0.shape[0], Q0.shape[0]), '',
         '## 1. How anisotropic is it?', '',
         '| measure | value | isotropic reference |', '|---|---|---|']
    Mu = unit_rows(M0)
    rng = np.random.default_rng(20260730)
    ia = rng.integers(0, Mu.shape[0], 40000)
    ib = rng.integers(0, Mu.shape[0], 40000)
    pair_raw = np.einsum('ij,ij->i', Mu[ia], Mu[ib])
    L += ['| mean cosine, random node pairs | **%.4f** | ~0.00 |'
          % pair_raw.mean(),
          '| sigma of that | %.4f | — |' % pair_raw.std(),
          '| \\|\\|mean vector\\|\\| / mean \\|\\|v\\|\\| | **%.4f** | ~0.00 |'
          % (np.linalg.norm(mu_d) / np.linalg.norm(M0, axis=1).mean()),
          '| PC1 variance share | **%.1f%%** | ~%.2f%% |'
          % (100 * var_share[0], 100.0 / M0.shape[1]),
          '| PC1-8 variance share | %.1f%% | ~%.2f%% |'
          % (100 * var_share[:8].sum(), 800.0 / M0.shape[1]),
          '| participation ratio (effective dims) | **%.0f** | 768 |'
          % part_ratio, '',
          '## 2. What does the dominant direction encode?', '',
          '- PC1 projection vs log(content length): r = **%+.3f**' % r_len, '']
    by_t = {}
    for s, p in zip(row_ids, proj1):
        by_t.setdefault(typ.get(s, '?'), []).append(p)
    top = sorted(by_t.items(), key=lambda kv: -len(kv[1]))[:8]
    L += ['| node type | n | mean PC1 projection |', '|---|---|---|']
    for t_, v in top:
        L.append('| %s | %d | %+.3f |' % (t_, len(v), float(np.mean(v))))

    # gold rows (with the raw turn timestamp) — needed inside the arm loop now
    gold_rows = []
    for t in turns:
        k = t['key'].split('/')
        key = (k[0], int(k[1]), int(k[2]))
        qi = qmap.get(key)
        gi = rid.get(master[int(t['gr'])])
        if qi is None or gi is None:
            continue
        gold_rows.append((qi, gi, ts_of.get(key)))

    # ── per-arm space metrics ──
    rows = []
    base_top = None
    for name, k in ARMS:
        if k < 0:
            Md, Qd = M0.copy(), Q0.copy()
        else:
            Md, Qd = M0 - mu_d, Q0 - mu_q
            for j in range(k):
                v = Vt[j]
                Md -= np.outer(Md @ v, v)
                Qd -= np.outer(Qd @ v, v)
        Mn, Qn = unit_rows(Md), unit_rows(Qd)
        pair = np.einsum('ij,ij->i', Mn[ia], Mn[ib])
        sims = Qn @ Mn.T                     # [queries x nodes]
        srt = np.sort(sims, axis=1)
        head = srt[:, -1] - srt[:, -TOPK]
        tops = np.argsort(-sims, axis=1)[:, :TOPK]
        if base_top is None:
            base_top = tops
            churn = 1.0
        else:
            churn = float(np.mean([
                len(set(a) & set(b_)) / TOPK for a, b_ in zip(tops, base_top)]))
        # Gold ranks are computed HERE, as-of masked, so the 245MB sims matrix
        # can be dropped before the next arm allocates its own (retaining one
        # per arm held ~1.5GB).
        ranks = []
        for qi, gi, ts in gold_rows:
            row = sims[qi]
            if ts:
                row = np.where(created <= ts, row, -np.inf)
            r = tie_fair(np.where(np.isfinite(row), row, np.nan), gi)
            if r is not None:
                ranks.append(r)
        rows.append({'name': name, 'pair_mean': pair.mean(),
                     'q_mean': sims.mean(), 'q_std': sims.std(),
                     'head': float(np.median(head)), 'churn': churn,
                     'ranks': ranks})
        del sims, srt, Md, Qd, Mn, Qn
    L += ['', '## 3. Dynamic range (label-free) + 4. churn vs raw', '',
          '| arm | random-pair cos | query→node cos mean | sigma | '
          'head spread (cos@1−cos@25) | top-25 overlap w/ raw |',
          '|---|---|---|---|---|---|']
    for r in rows:
        L.append('| %s | %.4f | %.4f | **%.4f** | **%.4f** | %.0f%% |'
                 % (r['name'], r['pair_mean'], r['q_mean'], r['q_std'],
                    r['head'], 100 * r['churn']))

    # ── 5. label-dependent: gold rank (as-of masked, computed in the loop) ──
    L += ['', '## 5. LABEL-DEPENDENT (secondary — depends on corpus-v2 gold)',
          '', 'n=%d turns with a gold in this view · nodes created after each '
          'turn are masked out (as-of honest)' % len(gold_rows), '',
          '| arm | median gold rank | @5 | @25 |', '|---|---|---|---|']
    for r in rows:
        ranks = r['ranks']
        L.append('| %s | %.0f | %.1f%% | %.1f%% |'
                 % (r['name'], float(np.median(ranks)),
                    100.0 * sum(1 for x in ranks if x <= 5) / len(ranks),
                    100.0 * sum(1 for x in ranks if x <= TOPK) / len(ranks)))
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
