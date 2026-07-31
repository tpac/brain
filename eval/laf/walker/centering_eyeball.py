"""Eyeball package — raw maxsim vs per-view-CENTRED maxsim, side by side.

The corpus cannot adjudicate centering: its golds were minted from pools that
raw cosine produced, so any geometry change looks bad against them (the @25
drop with an intact @5 is that signature). The only non-circular judge is the
operator reading both lists.

INTERVENTION UNDER TEST — the real candidate, not the diagnostic:
every embedding view is centred by ITS OWN mean (queries by the query mean),
then maxsim is recomputed. That addresses BOTH findings at once: the space-wide
83%-length common component (anisotropy) and the 0.086 between-view mean spread
(the aggregation defect), because centring each view removes its own offset.

Output: markdown, N random turns (seeded — NOT cherry-picked for drama), each
showing the operator message, the corpus gold, and the two top-10 lists with
membership markers so divergence is readable at a glance. The operator judges
two things: which list better serves the moment, AND whether the corpus gold
was even the right answer.

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 OMP_NUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/centering_eyeball.py
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
from servers.recall_laf import LafV1Engine, MAXSIM_VIEWS            # noqa: E402

REPORT = OUT_DIR / 'centering_eyeball.md'
N_TURNS = 14
TOPN = 10


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def rank_of(sims, gi):
    return int((sims > sims[gi]).sum()) + 1


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    w = open_walker()
    qv = {}
    for sess, epoch, seq, q in w.execute(
            'SELECT session_id, epoch, seq, q_vec FROM turns'):
        if q:
            qv[(sess, epoch, seq)] = np.frombuffer(
                q, dtype=np.float32).astype(np.float64)
    w.close()

    b = open_brain_ro()
    meta = {nid: (t or '?', ti or '', (c or 0))
            for nid, t, ti, c in b.execute(
                'SELECT id, type, title, LENGTH(content) FROM nodes')}
    b.close()

    turns, _e, _n = D.build_corpus('2026-05-11')
    rng = np.random.default_rng(20260730)
    picked = [turns[i] for i in rng.permutation(len(turns))[:N_TURNS * 3]]

    from tests.isolated_brain import IsolatedBrain
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
        n = eng._n
        ids = list(eng._master[:n])
        rid = {s: i for i, s in enumerate(ids)}

        raw, cen, mu = {}, {}, {}
        for v in MAXSIM_VIEWS:
            M = np.asarray(eng._mats[v][:n], dtype=np.float64)
            ok = np.isfinite(M).all(axis=1)
            mu[v] = M[ok].mean(axis=0)
            raw[v] = (unit_rows(M), ok)
            Mc = M - mu[v]
            cen[v] = (unit_rows(Mc), ok)
        # query-side mean, per view (each view's query offset is its own)
        allq = np.vstack([qv[k] for k in list(qv)[:4000]])
        muq = allq.mean(axis=0)

        out, used = [], 0
        for t in picked:
            if used >= N_TURNS:
                break
            k = t['key'].split('/')
            q = qv.get((k[0], int(k[1]), int(k[2])))
            gi = rid.get(master[int(t['gr'])])
            if q is None or gi is None:
                continue
            qn = q / np.linalg.norm(q)
            qc = q - muq
            qc = qc / np.linalg.norm(qc)

            def score(bank, query):
                cols = []
                for v in MAXSIM_VIEWS:
                    Mn, ok = bank[v]
                    s = Mn @ query
                    cols.append(np.where(ok, s, -np.inf))
                return np.max(np.stack(cols), axis=0)

            s_raw = score(raw, qn)
            s_cen = score(cen, qc)
            if not np.isfinite(s_raw[gi]) or not np.isfinite(s_cen[gi]):
                continue
            used += 1
            o_raw = np.argsort(-s_raw)[:TOPN]
            o_cen = np.argsort(-s_cen)[:TOPN]
            bd = bundles.get(t['key']) or {}
            out.append({
                'key': t['key'], 'stratum': t['stratum'],
                'cue': (bd.get('op_text') or '')[:420].replace('\n', ' '),
                'gold': master[int(t['gr'])][:8],
                'r_raw': rank_of(s_raw, gi), 'r_cen': rank_of(s_cen, gi),
                'raw': [ids[i][:8] for i in o_raw],
                'cen': [ids[i][:8] for i in o_cen],
                'overlap': len(set(o_raw) & set(o_cen)),
            })

    def line(sid, other, gold):
        ty, ti, _c = meta.get(
            next((x for x in meta if x[:8] == sid), ''), ('?', sid, 0))
        mark = '**★GOLD**' if sid == gold else ('=' if sid in other else '+')
        return '%s `%s` [%s] %s' % (mark, sid, ty, ti[:78])

    L = ['# Centering eyeball — raw maxsim vs per-view-centred maxsim', '',
         'Every view centred by its own mean; queries centred by the query '
         'mean; maxsim recomputed. %d turns, randomly sampled (seed 20260730 — '
         'not selected for drama).' % len(out), '',
         'Markers: **★GOLD** = the corpus gold · `=` also in the other list · '
         '`+` only in this list. Two questions worth judging: **which list '
         'better serves the moment**, and **was the corpus gold even right**.',
         '', '---', '']
    for i, r in enumerate(out, 1):
        L += ['### %d. [%s] gold rank: raw **%d** → centred **%d** · top-10 '
              'overlap %d/10' % (i, r['stratum'], r['r_raw'], r['r_cen'],
                                 r['overlap']), '',
              '> %s' % (r['cue'] or '(empty)'), '',
              '| # | RAW maxsim | CENTRED maxsim |', '|---|---|---|']
        for j in range(TOPN):
            a = line(r['raw'][j], set(r['cen']), r['gold'])
            b_ = line(r['cen'][j], set(r['raw']), r['gold'])
            L.append('| %d | %s | %s |' % (j + 1, a, b_))
        L += ['']
    n_better = sum(1 for r in out if r['r_cen'] < r['r_raw'])
    n_worse = sum(1 for r in out if r['r_cen'] > r['r_raw'])
    L += ['---', '',
          '## Summary over this sample', '',
          '- gold rank improved by centring: **%d** · worsened: **%d** · '
          'unchanged: %d' % (n_better, n_worse, len(out) - n_better - n_worse),
          '- mean top-10 overlap: %.1f/10' % np.mean([r['overlap'] for r in out]),
          '', '(Sample of %d is for READING, not for deciding — the aggregate '
          'numbers live in anisotropy_diagnostic.md.)' % len(out)]
    REPORT.write_text('\n'.join(L) + '\n')
    print('wrote %s — %d turns, gold better %d / worse %d'
          % (REPORT.name, len(out), n_better, n_worse))
    return 0


if __name__ == '__main__':
    sys.exit(main())
