"""Graph-lane (design B) noise refinements — measured on rescue vs noise.

Seeds = MAXSIM-lane top-5 (the stable, LAF-independent seeding Tom chose).
Per clean valid MISS, every time-honest 1-hop NEIGHBOR of the seeds is a row:
rescue (neighbor == gold) or noise. Four refinement axes tested:

  R1  edge-why cosine — cos(q_vec, edge_relations.embedding), best edge per
      neighbor (the rebuild-spec scoring). Separation + threshold curve.
  R2  convergence — neighbor reached from ≥2 distinct seeds (Tom's
      convergence-first principle, measured at 1 hop).
  R3  co_accessed strength — co_access_count / recency of last_strengthened
      for the behavioral channel. CAVEAT (review 2026-07-24): co_access_count
      is the PRESENT-DAY cumulative counter, not time-gated to the turn — a
      time-leaky feature. It showed NO separation anyway (axis dead) and the
      count-based filter was rejected from the final spec, so no conclusion
      rests on it; do not resurrect this axis without a time-gated recount.
  R4  target priors — neighbor node type / content size.

Output: per-axis separation + a CUMULATIVE filter curve (rescues kept vs
fan-out kept vs nodes-added-per-turn) ending in the refined walk spec.

Read-only. Run: ./dev python3 eval/laf/walker/corpus_v2_hop_refine.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, WALKER_DB, open_ro, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn                                          # noqa: E402
from layer_readout_probe import lane_z                              # noqa: E402

CUTOFF = '2026-05-11'
K = 5
GENERIC = ('related_to', 'related', 'community_member')
REPORT = OUT_DIR / 'corpus_v2_hop_refine.md'


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    lanes_mm = np.load(OUT_DIR / 'lane_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    master = idx['master']
    m2i = {sid: i for i, sid in enumerate(master)}
    n = idx['n_nodes']
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    # q_vec per labeled turn
    w = open_ro(WALKER_DB)
    qvec = {}
    for sess, epoch, seq, qv in w.execute(
            'SELECT session_id, epoch, seq, q_vec FROM turns WHERE labeled=1'):
        if qv is not None:
            qvec[(sess, epoch, seq)] = np.frombuffer(qv, dtype=np.float32)
    w.close()

    # graph with embeddings + node meta
    b = open_brain_ro()
    node_meta = {}
    for nid, typ, clen in b.execute(
            'SELECT id, type, LENGTH(content) FROM nodes'):
        if nid in m2i:
            node_meta[m2i[nid]] = (typ, clen or 0)
    adj = defaultdict(list)
    for src, tgt, rel, dlen, ecr, emb, cac, lstr in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, "
            "LENGTH(COALESCE(r.description,'')), e.created_at, r.embedding, "
            "e.co_access_count, e.last_strengthened "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        ev = np.frombuffer(emb, dtype=np.float32) if emb is not None else None
        rec = (rel, dlen or 0, ecr, ev, cac or 0, lstr)
        adj[si].append((ti,) + rec)
        adj[ti].append((si,) + rec)
    b.close()

    rows = []          # neighbor-level rows
    n_miss = 0
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        bd = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not bd or (bd['ts'] or '') < CUTOFF:
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.mh is None or np.isnan(tt.fields[0]).all():
            continue
        mix = 0.65 * zn(tt.fields[0]) + 0.35 * zn(tt.mh)
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        if int(np.where(np.argsort(-fin) == tt.gr)[0][0]) + 1 <= 5:
            continue
        n_miss += 1
        turn_dt = iso(bd['ts'])
        qv = qvec.get(tuple(t['key']))
        L = lanes_mm[t['row']].astype(np.float32)
        raw_mx = L[S['op0'], 0]
        mxz = lane_z(raw_mx, 'maxsim', np.isfinite(raw_mx), n)
        mfin = np.where(np.isfinite(mxz), mxz, -np.inf)
        seeds = [int(x) for x in np.argsort(-mfin)[:K]]
        neigh = {}         # oi -> aggregated
        for si in seeds:
            for (oi, rel, dlen, ecr, ev, cac, lstr) in adj.get(si, ()):
                edt = iso(ecr)
                if turn_dt and edt and edt > turn_dt:
                    continue
                cos = float(qv @ ev) if (qv is not None and ev is not None) \
                    else None
                d = neigh.setdefault(oi, {
                    'seeds': set(), 'best_cos': None, 'best_rel': None,
                    'best_dlen': 0, 'has_coacc': False, 'coacc_n': 0,
                    'coacc_fresh_d': None, 'has_sem': False})
                d['seeds'].add(si)
                if rel == 'co_accessed':
                    d['has_coacc'] = True
                    d['coacc_n'] = max(d['coacc_n'], cac)
                    ls = iso(lstr)
                    if turn_dt and ls and ls <= turn_dt:
                        fd = (turn_dt - ls).days
                        if d['coacc_fresh_d'] is None or fd < d['coacc_fresh_d']:
                            d['coacc_fresh_d'] = fd
                elif rel not in GENERIC:
                    d['has_sem'] = True
                    if cos is not None and (d['best_cos'] is None
                                            or cos > d['best_cos']):
                        d['best_cos'] = cos
                        d['best_rel'] = rel
                    if dlen > d['best_dlen']:
                        d['best_dlen'] = dlen
        for oi, d in neigh.items():
            nm = node_meta.get(oi, (None, 0))
            rows.append({'rescue': oi == tt.gr, 'n_seeds': len(d['seeds']),
                         **{k: d[k] for k in ('best_cos', 'best_dlen',
                                              'has_coacc', 'coacc_n',
                                              'coacc_fresh_d', 'has_sem')},
                         'tgt_type': nm[0], 'tgt_size': nm[1]})

    R = [r for r in rows if r['rescue']]
    Nz = [r for r in rows if not r['rescue']]
    L = ['# Graph-lane refinements — rescue vs noise (maxsim-top5 seeds, '
         'neighbor-level)', '',
         'misses: %d · neighbor rows: %d (rescue %d · noise %d) · blind '
         'fan-out %.1f nodes/turn'
         % (n_miss, len(rows), len(R), len(Nz), len(Nz) / max(1, n_miss)), '']

    def med(a):
        a = [x for x in a if x is not None]
        return float(np.median(a)) if a else float('nan')

    # R1 — edge-why cosine
    rc = [r['best_cos'] for r in R if r['best_cos'] is not None]
    nc = [r['best_cos'] for r in Nz if r['best_cos'] is not None]
    L += ['## R1. Edge-why cosine (semantic edges w/ embedding)', '',
          '| group | n | median cos | p25 | p75 |', '|---|---|---|---|---|',
          '| rescue | %d | %.3f | %.3f | %.3f |'
          % (len(rc), med(rc), np.percentile(rc, 25), np.percentile(rc, 75)),
          '| noise | %d | %.3f | %.3f | %.3f |'
          % (len(nc), med(nc), np.percentile(nc, 25), np.percentile(nc, 75)),
          '', '| cos ≥ τ | rescues kept | noise kept |', '|---|---|---|']
    for tau in (0.30, 0.35, 0.40, 0.45, 0.50):
        rk = 100 * np.mean([c >= tau for c in rc]) if rc else 0
        nk = 100 * np.mean([c >= tau for c in nc]) if nc else 0
        L.append('| %.2f | %.0f%% | %.0f%% |' % (tau, rk, nk))
    L.append('')

    # R2 — convergence
    L += ['## R2. Convergence (reached from ≥2 seeds)', '',
          '| n_seeds | rescue rate | share of rows |', '|---|---|---|']
    for ns in (1, 2, 3):
        sub = [r for r in rows if (r['n_seeds'] >= ns if ns > 1
                                   else r['n_seeds'] == 1)]
        if sub:
            L.append('| %s | %.1f%% | %.0f%% |'
                     % ('=1' if ns == 1 else '≥%d' % ns,
                        100 * np.mean([r['rescue'] for r in sub]),
                        100 * len(sub) / len(rows)))
    L.append('')

    # R3 — co_accessed strength
    co_r = [r for r in R if r['has_coacc']]
    co_n = [r for r in Nz if r['has_coacc']]
    L += ['## R3. co_accessed channel', '',
          '- rescue rows via co_accessed: %d · noise: %d' % (len(co_r), len(co_n)),
          '- co_access_count median: rescue %.0f vs noise %.0f'
          % (med([r['coacc_n'] for r in co_r]),
             med([r['coacc_n'] for r in co_n])),
          '- days since last_strengthened: rescue %.0f vs noise %.0f'
          % (med([r['coacc_fresh_d'] for r in co_r]),
             med([r['coacc_fresh_d'] for r in co_n])), '']

    # R4 — target priors
    tr = Counter(r['tgt_type'] for r in R)
    tn = Counter(r['tgt_type'] for r in Nz)
    L += ['## R4. Target-node priors', '',
          '- rescue sizes median %.0f vs noise %.0f'
          % (med([r['tgt_size'] for r in R]), med([r['tgt_size'] for r in Nz])),
          '- noise-heavy target types (noise%%/rescue%%): ' + ', '.join(
              '%s %.0f%%/%.0f%%' % (ty, 100 * tn[ty] / len(Nz),
                                    100 * tr.get(ty, 0) / len(R))
              for ty, _ in tn.most_common(6)), '']

    # CUMULATIVE filter curve
    def curve(name, keep):
        rk = sum(1 for r in R if keep(r))
        nk = sum(1 for r in Nz if keep(r))
        return ('| %s | %.0f%% (%d/%d) | %.0f%% | %.1f |'
                % (name, 100 * rk / max(1, len(R)), rk, len(R),
                   100 * nk / max(1, len(Nz)), nk / max(1, n_miss)))
    L += ['## Cumulative filter curve (neighbor kept if ANY channel passes)',
          '', '| walk spec | rescues kept | noise kept | noise nodes/turn |',
          '|---|---|---|---|',
          curve('blind +1hop (all edges)', lambda r: True),
          curve('base union: co_acc ∪ sem(desc≥80)',
                lambda r: r['has_coacc'] or (r['has_sem'] and r['best_dlen'] >= 80)),
          curve('+ why-cos≥0.40 on sem channel',
                lambda r: r['has_coacc'] or (r['has_sem'] and r['best_dlen'] >= 80
                                             and (r['best_cos'] or 0) >= 0.40)),
          curve('+ co_acc needs fresh≤14d OR n≥2',
                lambda r: (r['has_coacc'] and ((r['coacc_fresh_d'] is not None
                                                and r['coacc_fresh_d'] <= 14)
                                               or r['coacc_n'] >= 2))
                or (r['has_sem'] and r['best_dlen'] >= 80
                    and (r['best_cos'] or 0) >= 0.40)),
          curve('+ convergence override (any edge, ≥2 seeds)',
                lambda r: r['n_seeds'] >= 2
                or (r['has_coacc'] and ((r['coacc_fresh_d'] is not None
                                         and r['coacc_fresh_d'] <= 14)
                                        or r['coacc_n'] >= 2))
                or (r['has_sem'] and r['best_dlen'] >= 80
                    and (r['best_cos'] or 0) >= 0.40)),
          '']
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
