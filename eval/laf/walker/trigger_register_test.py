"""Does TRIGGER-REGISTER situation text rescue the untouched golds?

The 18% of misses no lane touches are dominated by action→precondition memories
whose `situation` field names the moment the lesson was LEARNED (often a
symptom) rather than the CLASS of moments where it applies. The slot and the
lane both already exist (`sit` = cosine(query, node _situation vector), and the
embedder already applies asymmetric nomic prefixes), so the hypothesis is that
this is a CONTENT problem, not an architecture problem.

ARMS (rewrites produced by a subagent that never saw the cues — a rewrite
written while looking at the failing message is hand-fitted to the test):
  A baseline    the situation text as encoded today
  B trigger     trigger-register rewrite: the class of triggering moments in
                operator register, including terse utterance forms
  C paraphrase  NEGATIVE CONTROL — same register, same scope, reworded. If C
                gains as much as B, the effect is "text was rewritten", not
                "register changed".

LENGTH CONFOUND: B is longer than A/C by construction (richer trigger classes
cost words). So B's gain is also checked for correlation with its length delta
— if short rewrites gain as much as long ones, length is not the driver.

Scoring: substitute ONLY the gold's raw sit value with cosine(q_vec, embedded
rewrite), re-znorm the lane, recompose f0 at shipped gains, remix with M_h at
λ=0.65, re-rank tie-fair. Reports sit-lane rank (does it enter sit's top-25 →
the T2 'organic' threshold, which also makes it hop-reachable) and mix rank.

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/trigger_register_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker, open_brain_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from edge_fusion_census import LAM, iso, verb_class                 # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from servers.recall_laf import zscore_variant                       # noqa: E402
from servers import embedder                                        # noqa: E402

REPORT = OUT_DIR / 'trigger_register_test.md'
SEED_K = 25
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc', 'mh')


def unit(buf):
    v = np.frombuffer(buf, dtype=np.float32).astype(np.float64)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def tie_fair_rank(scores, gi):
    gv = scores[gi]
    if not np.isfinite(gv):
        return None
    fin = np.where(np.isfinite(scores), scores, -np.inf)
    return int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1


def main():
    rew = {x['id']: x for x in
           json.loads((OUT_DIR / 'trigger_rewrites.json').read_text())}
    src = {x['id']: x for x in
           json.loads((OUT_DIR / 'untouched_golds.json').read_text())}
    print('rewrites: %d' % len(rew))

    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']
    m2i = {sid: i for i, sid in enumerate(master)}
    n_nodes = idx['n_nodes']

    # embed the two rewrite arms (document side, like stored situations)
    embedder.load_model()
    texts, keys = [], []
    for sid, r in rew.items():
        for arm in ('trigger', 'paraphrase'):
            texts.append(r[arm])
            keys.append((sid, arm))
        # ARM A' — STALE-EMBEDDING CONTROL. Freshly embed the UNCHANGED
        # situation text. If A' ≫ A(stored lane value), the baseline is a
        # stale `_situation` vector (text revised without re-embedding) and
        # every rewrite arm's gain is an artifact of re-embedding, not of
        # register. This control gates the entire result.
        texts.append(src[sid]['situation'] or src[sid]['title'])
        keys.append((sid, 'refresh'))
    blobs = embedder.embed_batch(texts, kind='document')
    vecs = {k: unit(b) for k, b in zip(keys, blobs) if b}
    print('embedded %d rewrite texts' % len(vecs))

    # q_vec per turn from the walker
    w = open_walker()
    qv = {}
    for sess, epoch, seq, q in w.execute(
            'SELECT session_id, epoch, seq, q_vec FROM turns'):
        if q:
            qv[(sess, epoch, seq)] = unit(q)
    w.close()

    # lane cache + slot index are loop-INVARIANT — open once, not per turn.
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    sit_li = A.LANES.index('sit')

    turns, _enr, n = D.build_corpus('2026-05-11')
    gains = np.array([A.GAINS[ln] for ln in A.LANES])
    rows = []

    for t in turns:
        U = np.flatnonzero(t['alive'])
        if U.size < 20:
            continue
        gid = master[int(t['gr'])][:8]
        if gid not in rew:
            continue
        key = tuple(t['key'].split('/'))
        key = (key[0], int(key[1]), int(key[2]))
        q = qv.get(key)
        if q is None:
            continue
        zl = {ln: t['zl'][ln] for ln in A.LANES}
        # raw sit values are not carried on the turn dict → recover the lane's
        # raw vector from the lane cache
        raw = lanes_mm[t['row_idx']][S['op0'], sit_li].astype(np.float64)
        gi = int(t['gr'])
        tdt = t.get('turn_dt')

        def score(new_raw_val):
            r2 = raw.copy()
            if new_raw_val is not None:
                r2[gi] = new_raw_val
            alive = t['alive']
            zsit = zscore_variant(r2, n_nodes, mask=alive, kind='current')
            Z = np.column_stack([(zsit if ln == 'sit' else zl[ln])[U]
                                 for ln in A.LANES])
            f0 = Z @ gains
            if not np.isfinite(f0).any() or f0.std() <= 1e-9:
                return None, None
            zf0 = (f0 - f0.mean()) / f0.std()
            mix = LAM * zf0 + (1.0 - LAM) * zn(t['mh'])[U]
            gpos = int(np.flatnonzero(U == gi)[0])
            sit_rank = tie_fair_rank(r2[U], gpos)
            return tie_fair_rank(mix, gpos), sit_rank

        base_mix, base_sit = score(None)
        if base_mix is None:
            continue
        rec = {'gid': gid, 'base_mix': base_mix, 'base_sit': base_sit,
               'title': src[gid]['title'][:70]}
        for arm in ('trigger', 'paraphrase', 'refresh'):
            v = vecs.get((gid, arm))
            if v is None:
                continue
            m, s = score(float(np.dot(q, v)))
            rec['%s_mix' % arm] = m
            rec['%s_sit' % arm] = s
            rec['%s_len' % arm] = len(
                rew[gid][arm] if arm in rew[gid]
                else (src[gid]['situation'] or src[gid]['title']))
        # would sit's top-25 now reach it, organically or by one hop?
        for arm in ('base', 'trigger', 'paraphrase', 'refresh'):
            sr = rec.get('%s_sit' % arm) if arm != 'base' else base_sit
            if sr is None:
                continue
            rec['%s_organic25' % arm] = sr <= SEED_K
        rows.append(rec)

    def med(vals):
        v = sorted(x for x in vals if x is not None)
        return v[len(v) // 2] if v else float('nan')

    L = ['# Trigger-register test — can better `situation` text rescue the '
         'untouched golds?', '',
         'n=%d untouched golds with rewrites, q_vec and a rankable turn. '
         'Rewrites written blind to the failing cue. Arm C (paraphrase) is the '
         'same-register control.' % len(rows), '',
         '| arm | median sit-lane rank | in sit top-25 | median mix rank | '
         'mix ≤25 | mix ≤5 | median chars |', '|---|---|---|---|---|---|---|']
    for arm, label in (('base', 'A baseline (STORED vector)'),
                       ('refresh', "A' same text, RE-EMBEDDED (stale check)"),
                       ('trigger', 'B trigger-register'),
                       ('paraphrase', 'C paraphrase (control)')):
        sk = 'base_sit' if arm == 'base' else '%s_sit' % arm
        mk = 'base_mix' if arm == 'base' else '%s_mix' % arm
        sits = [r.get(sk) for r in rows]
        mixes = [r.get(mk) for r in rows]
        n25 = sum(1 for r in rows if r.get('%s_organic25' % arm))
        m25 = sum(1 for x in mixes if x is not None and x <= 25)
        m5 = sum(1 for x in mixes if x is not None and x <= 5)
        lens = ([len(src[r['gid']]['situation']) for r in rows] if arm == 'base'
                else [r.get('%s_len' % arm, 0) for r in rows])
        L.append('| %s | %.0f | %d (%.0f%%) | %.0f | %d | %d | %d |'
                 % (label, med(sits), n25, 100.0 * n25 / max(len(rows), 1),
                    med(mixes), m25, m5, med(lens)))

    # paired deltas + the length-confound check
    L += ['', '## Paired deltas vs baseline (sit-lane rank; negative = better)',
          '', '| arm | median Δ | improved | worsened | rank-corr(Δ, length) |',
          '|---|---|---|---|---|']
    for arm in ('refresh', 'trigger', 'paraphrase'):
        d, dl = [], []
        for r in rows:
            a, b_ = r.get('base_sit'), r.get('%s_sit' % arm)
            if a is None or b_ is None:
                continue
            d.append(b_ - a)
            dl.append(r.get('%s_len' % arm, 0))
        if not d:
            continue
        d = np.array(d, dtype=float)
        dl = np.array(dl, dtype=float)
        # Spearman via rank-transform (no scipy in the bundled env)
        rd = np.argsort(np.argsort(d))
        rl = np.argsort(np.argsort(dl))
        corr = float(np.corrcoef(rd, rl)[0, 1]) if len(d) > 3 else float('nan')
        L.append('| %s | %+.0f | %d | %d | %+.2f |'
                 % (arm, np.median(d), int((d < 0).sum()), int((d > 0).sum()),
                    corr))
    L += ['', '(rank-corr near 0 ⇒ longer rewrites did not gain more, so the '
          'effect is register rather than length.)', '',
          '## Biggest movers (sit-lane rank, baseline → trigger)', '']
    movers = sorted((r for r in rows if r.get('trigger_sit') is not None),
                    key=lambda r: (r['trigger_sit'] - r['base_sit']))[:12]
    for r in movers:
        L.append('- `%s` %s → **%s** (mix %s → %s) — %s'
                 % (r['gid'], r['base_sit'], r['trigger_sit'],
                    r['base_mix'], r.get('trigger_mix'), r['title']))
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
