"""Door-SEPARATED evaluation — the exam each mechanism is actually sitting.

THE ERROR THIS FIXES. Door-1 and door-2 are DEDICATED corpora, not slices of
one: `cue` golds exam contextless recall (door-1); `window`/`session` golds
exam conversation-state carry (door-2, the Moment/running-field business).
The rubric says so (corpus_v2_judge_prompt "Strata are exam doors"), 449fb9a7
says "when evaluating door-1 changes use CUE-SUFF only", and d0d82551 defines
the two exam populations. Every arm in this arc was nonetheless FIT AND SCORED
on the blend, which does two bad things at once:
  * it averages two different exams, so a door-1 mechanism's gain is diluted
    (or cancelled) by its door-2 cost -- exactly what happened to enrichment
    (cue +2.9/+4.1pp, window -2.6/-4.0pp, blend ~0);
  * the GAINS themselves were fit on the mixture, so no arm was ever tuned
    for the exam it targets.

Reporting per stratum afterwards does NOT fix it. The gains have to be fit on
the door's own population. That is what this does.

DOORS
  door-1  cue                    -- contextless recall
  door-2  window + session       -- conversation-state carry

PRE-REGISTERED. Same protocol as laf_confirm: 3 arms (shipped / refit-5 /
refit-5 + enrichment K=20+corridors), >=3 session->fold permutations
(9ca6cd5b: a CI inside ONE partition is blind to partition variance), 2
corpora, paired bootstrap. Pass = CI excludes 0 across every seed.

POWER, stated up front so a null is readable: door-1 quality n~306, so the
paired sd rises ~sqrt(707/306)=1.4x to ~1.4pp and MDE(95%) ~2.9pp. Enrichment's
cue effect measured +2.9 to +4.1pp on the blend-fit gains, so door-1 is the
one place in this arc where the effect may exceed its own detection floor.
Smaller n, but a bigger and un-cancelled effect.

SCOPE LIMIT ON THE lambda RESULT (review 2026-07-25 — must travel with the
claim). A.build() filters to turns rankable under the PRODUCTION mix, i.e.
lambda=0.65 (793 built -> 707 kept, 11% dropped). Turns with no usable history
are therefore excluded UPSTREAM of this fit -- and those are exactly the turns
where lambda=1.0 would matter most (nothing to blend). Measured on what
survives: 0/707 golds have non-finite zn(mh), so the arms here do share one
ranking universe and the comparison is clean. But "the fit never chose
lambda=1.0" is scoped to TURNS THAT HAVE USABLE HISTORY; it says nothing about
history-less turns (session openers). Testing those needs a corpus built
without the lambda=0.65 rankability filter.

(The 0*NaN hazard this check was looking for is real in principle -- at
lambda<1.0 a NaN zmh would poison mix and silently shrink the universe
relative to lambda=1.0 -- but it does not fire on this corpus, and the
`else zf0` branch in rank_lam guards the lambda=1.0 end regardless.)

Read-only. Run:  ./dev python3 eval/laf/walker/laf_doors.py
"""
import json
import sys

import numpy as np

from walker_db import OUT_DIR, WALKER_DB, open_ro, open_brain_ro

REPO = __file__.rsplit('/eval/', 1)[0]
sys.path.insert(0, REPO)
sys.path.append(str(OUT_DIR))
from lambda_probe import zn                                         # noqa: E402
import enrichment_lane as EL                                       # noqa: E402
import laf_lane_audit as A                                         # noqa: E402
import laf_real_perf as RP                                         # noqa: E402
from enrichment_widen import load_communities                      # noqa: E402

FOLD_SEEDS = (0, 1, 2, 3, 4)
FOLDS = 5
DOORS = (('door-1 (cue) — contextless recall', {'cue'}),
         ('door-2 (window+session) — conversation-state carry',
          {'window', 'session'}))

# λ = weight on the CURRENT-message field; (1−λ) goes to the moment/history.
# Production pins λ=0.65 and every measurement in this arc inherited it — on
# BOTH doors. That makes door-1 incoherent: its exam is contextless recall
# ("would this node be warranted if the message opened a brand-new session?")
# yet the scorer was handed 35% conversation history to answer it. So λ is a
# FITTED PER-DOOR parameter here, and λ=1.0 (moment fully OFF) is in the grid
# as the honest door-1 configuration. This also re-opens 137302a6's "don't
# retune λ" — that verdict was measured on the blend, so per 312021a2 it does
# not carry over to a per-door fit.
LAM_GRID = (1.0, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2, 0.0)
REPORT = OUT_DIR / 'laf_doors.md'


def build_corpus(cutoff):
    """Turns + the enrichment variant, once per corpus."""
    old = A.CUTOFF
    A.CUTOFF = cutoff
    try:
        turns, n = A.build()
    finally:
        A.CUTOFF = old
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    row_of = {tuple(t['key']): t['row'] for t in idx['turns']}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}
    w = open_ro(WALKER_DB)
    qvecs = EL.build_qvecs(w)
    w.close()
    b = open_brain_ro()
    node_meta = EL.build_node_meta(b, m2i)
    adj = EL.build_adjacency(b, m2i)
    of_node, members, corridor, cohesion = load_communities(b, m2i)
    b.close()
    for t in turns:
        p = t['key'].split('/')
        kk = (p[0], int(p[1]), int(p[2]))
        t['row_idx'] = row_of[kk]
        t['qv'] = qvecs.get(kk)
        bd = bundles.get(t['key'])
        t['turn_dt'] = EL.iso(bd['ts']) if bd else None
    enr = [RP.enrichment_variant(t, adj, node_meta, lanes_mm, S, n, 20,
                                 of_node, members, corridor, cohesion)
           for t in turns]
    return turns, enr, n


ARMS = {'A shipped (λ=0.65 pinned)': (list(A.LANES), True),
        'B refit gains+λ': (list(A.LANES), False),
        'C refit gains+λ + enrichment': (list(A.LANES) + ['ENR'], False)}


def rank_lam(p, gvec, lam):
    """Gold's tie-fair rank under an explicit λ (RP.rank_of pins A.LAM)."""
    f0 = p['Z'] @ gvec
    if f0.size <= 2 or f0.std() <= 1e-9:
        return None
    zf0 = (f0 - f0.mean()) / f0.std()
    mix = lam * zf0 + (1.0 - lam) * p['zmh'] if lam < 1.0 else zf0
    gv = mix[p['g']]
    if not np.isfinite(gv):
        return None
    fin = np.where(np.isfinite(mix), mix, -np.inf)
    return int((fin > gv).sum()) + (int((fin == gv).sum()) - 1) / 2.0 + 1


def hits_lam(rows, gvec, lam, at=5):
    out = []
    for p in rows:
        r = rank_lam(p, gvec, lam)
        out.append(np.nan if r is None else float(r <= at))
    return np.array(out)


def refit_gains_lam(rows, lanes, passes=2):
    """Coordinate ascent over gains AND λ — λ is just one more coordinate."""
    best, bs = None, -1.0
    for g0, l0 in (([A.GAINS.get(ln, 0.5) for ln in lanes], 0.65),
                   ([1.25 if ln == 'maxsim' else 0.4 for ln in lanes], 1.0)):
        g, lam = np.array(g0, dtype=float), l0
        for _ in range(passes):
            for j in range(len(lanes)):
                bv, bsc = g[j], -1.0
                for c in RP.GRID:
                    cand = g.copy(); cand[j] = c
                    s = np.nanmean(hits_lam(rows, cand, lam))
                    if s > bsc:
                        bsc, bv = s, c
                g[j] = bv
            bl, bsc = lam, -1.0
            for c in LAM_GRID:
                s = np.nanmean(hits_lam(rows, g, c))
                if s > bsc:
                    bsc, bl = s, c
            lam = bl
        s = np.nanmean(hits_lam(rows, g, lam))
        if s > bs:
            bs, best = s, (g.copy(), lam)
    return best


def prep_rows(turns, enr, strata, lanes):
    rows = []
    for i, t in enumerate(turns):
        if t['stratum'] not in strata:
            continue
        U = np.flatnonzero(t['alive'])
        cols = [(enr[i] if ln == 'ENR' else t['zl'][ln])[U] for ln in lanes]
        gpos = int(np.searchsorted(U, t['gr']))
        if gpos >= len(U) or U[gpos] != t['gr']:
            continue
        rows.append({'Z': np.column_stack(cols).astype(np.float64),
                     'zmh': zn(t['mh'])[U], 'g': gpos, 'sess': t['sess']})
    return rows


def eval_door(turns, enr, strata):
    prep = {nm: (prep_rows(turns, enr, strata, lanes), lanes, fixed)
            for nm, (lanes, fixed) in ARMS.items()}
    lens = {k: len(v[0]) for k, v in prep.items()}
    if len(set(lens.values())) != 1:
        raise SystemExit('PAIRING BROKEN: %s' % lens)
    out, lams = {}, {}
    ref_key = next(iter(prep))
    for seed in FOLD_SEEDS:
        rng = np.random.default_rng(seed)
        sess = sorted({p['sess'] for p in prep[ref_key][0]})
        perm = rng.permutation(len(sess))
        fold_of = {sess[perm[i]]: i % FOLDS for i in range(len(sess))}
        for nm, (rows, lanes, fixed) in prep.items():
            fold = np.array([fold_of[p['sess']] for p in rows])
            H = np.full(len(rows), np.nan)
            if fixed:
                H = hits_lam(rows, np.array([A.GAINS[l] for l in lanes]), 0.65)
            else:
                seen = []
                for f in range(FOLDS):
                    tr = [rows[i] for i in range(len(rows)) if fold[i] != f]
                    te = [i for i in range(len(rows)) if fold[i] == f]
                    g, lam = refit_gains_lam(tr, lanes)   # THIS DOOR ONLY
                    seen.append(lam)
                    hh = hits_lam([rows[i] for i in te], g, lam)
                    for j, i in enumerate(te):
                        H[i] = hh[j]
                lams.setdefault(nm, []).extend(seen)
            out[(seed, nm)] = H
    return out, lens[ref_key], lams


def main():
    L = ['# Door-separated evaluation — each mechanism on its own exam', '',
         'Door-1 (`cue`) and door-2 (`window`+`session`) are DEDICATED corpora, '
         'not slices of one. Every earlier arm in this arc was fit AND scored '
         'on the blend, which diluted a door-1 mechanism by its door-2 cost '
         '(enrichment: cue +2.9/+4.1pp, window −2.6/−4.0pp, blend ≈0) and tuned '
         'the gains to a mixture of two exams. Here the gains are **fit on each '
         "door's own population**.", '',
         '5 session→fold permutations · 2 corpora · paired bootstrap ×4000 · '
         'pass = CI excludes 0 across every seed.', '']

    for clabel, cutoff in (('quality (≥2026-05-11)', '2026-05-11'),
                           ('wide (all valid golds)', '0000')):
        print('=== corpus %s ===' % clabel)
        turns, enr, n = build_corpus(cutoff)
        for dlabel, strata in DOORS:
            out, dn, lams = eval_door(turns, enr, strata)
            sd_ref = RP.boot(out[(0, 'B refit gains+λ')],
                             out[(0, 'A shipped (λ=0.65 pinned)')])[1]
            L += ['## %s · %s · n=%d' % (clabel, dlabel, dn), '',
                  '- paired sd ≈ %.2fpp → **MDE(95%%) ≈ %.1fpp** at this n'
                  % (sd_ref, 1.96 * sd_ref), '',
                  '| seed | A shipped | B refit | C +enrichment | B−A (95% CI) | C−A (95% CI) | C−B (95% CI) |',
                  '|---|---|---|---|---|---|---|']
            ba, ca, cb = [], [], []
            for seed in FOLD_SEEDS:
                HA = out[(seed, 'A shipped (λ=0.65 pinned)')]
                HB = out[(seed, 'B refit gains+λ')]
                HC = out[(seed, 'C refit gains+λ + enrichment')]
                m1, _, l1, h1 = RP.boot(HB, HA)
                m2, _, l2, h2 = RP.boot(HC, HA)
                m3, _, l3, h3 = RP.boot(HC, HB)
                ba.append(l1 > 0); ca.append(l2 > 0); cb.append(l3 > 0)
                L.append('| %d | %.1f%% | %.1f%% | %.1f%% | %+.2f [%+.2f, %+.2f] '
                         '| %+.2f [%+.2f, %+.2f] | %+.2f [%+.2f, %+.2f] |'
                         % (seed, 100*np.nanmean(HA), 100*np.nanmean(HB),
                            100*np.nanmean(HC), m1, l1, h1, m2, l2, h2,
                            m3, l3, h3))
            L += ['', '- **B−A excludes 0 in %d/%d seeds · C−A in %d/%d · '
                  'C−B in %d/%d**'
                  % (sum(ba), len(ba), sum(ca), len(ca), sum(cb), len(cb)), '',
                  '- **fitted λ** (weight on the current message; 1.0 = moment '
                  'OFF) — %s'
                  % ' · '.join(
                      '%s: %s' % (nm.split('(')[0].strip(),
                                  '/'.join('%.2f' % x for x in sorted(set(v))))
                      for nm, v in lams.items()), '']
            print('  %s n=%d: B-A %d/5, C-A %d/5, C-B %d/5'
                  % (dlabel, dn, sum(ba), sum(ca), sum(cb)))

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
