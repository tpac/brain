"""1-hop anatomy — does the RESCUING hop have a character the NOISE lacks?

Tom: '+1 hop everywhere brings nodes but also noise — does the rescue have a
character on node type, edge type, node size, date, target?'

On clean valid MISSES (static λ-mix rank>5, turn ≥ CUTOFF): seeds = top-5 of
the mix. A RESCUE edge = any active edge (either direction) between a seed
and the gold. NOISE = every other 1-hop edge those same seeds fan out to.
Profile rescue vs noise on: relation, edge-description length, edge age at
turn, seed rank, endpoint node type, endpoint content size, endpoint age,
endpoint degree (hub-ness). Then the FILTER CURVE: for candidate filters,
% rescues kept vs % fan-out cut — the selective-walk design table.

Time-honest: edges created AFTER the turn are excluded per turn (an edge
that didn't exist yet can neither rescue nor add noise).

Read-only brain.db + caches. Run: ./dev python3 eval/laf/walker/corpus_v2_hop_anatomy.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn                                          # noqa: E402

CUTOFF = '2026-05-11'
LAM = 0.65
K_SEEDS = 5
REPORT = OUT_DIR / 'corpus_v2_hop_anatomy.md'


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    master = idx['master']
    m2i = {sid: i for i, sid in enumerate(master)}
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    # ── graph: active edges + primary relation + node metadata ──
    b = open_brain_ro()
    node_meta = {}
    for nid, typ, created, clen in b.execute(
            'SELECT id, type, created_at, LENGTH(content) FROM nodes'):
        if nid in m2i:
            node_meta[m2i[nid]] = (typ, created, clen or 0)
    edges = []           # (src_i, tgt_i, relation, desc_len, created_at)
    deg = Counter()
    for src, tgt, rel, dlen, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, "
            "LENGTH(COALESCE(r.description,'')), e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        edges.append((si, ti, rel, dlen or 0, created))
        deg[si] += 1
        deg[ti] += 1
    b.close()
    adj = defaultdict(list)  # node_i -> [(other_i, rel, dlen, created)]
    for si, ti, rel, dlen, created in edges:
        adj[si].append((ti, rel, dlen, created))
        adj[ti].append((si, rel, dlen, created))
    print('graph: %d edge-relations over master, %d nodes with edges'
          % (len(edges), len(adj)))

    rescues, per_turn = [], []
    noise_rows = []          # ALIGNED (rel, desc_len, neighbor_deg) tuples
    noise_tgt_type = Counter()
    for t in idx['turns']:
        key = '%s/%d/%d' % tuple(t['key'])
        v = verds.get(key)
        bd = bundles.get(key)
        if not v or v['verdict'] != 'valid' or not bd or (bd['ts'] or '') < CUTOFF:
            continue
        tt = Turn(t, fields, S)
        if tt.gr < 0 or tt.mh is None or np.isnan(tt.fields[0]).all():
            continue
        mix = LAM * zn(tt.fields[0]) + (1 - LAM) * zn(tt.mh)
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        order = np.argsort(-fin)
        gold_rank = int(np.where(order == tt.gr)[0][0]) + 1
        if gold_rank <= 5:
            continue                                   # hits don't need hops
        turn_dt = iso(bd['ts'])
        seeds = [int(x) for x in order[:K_SEEDS]]
        fan = set()
        found = []
        for srank, si in enumerate(seeds, 1):
            for (oi, rel, dlen, created) in adj.get(si, ()):
                edt = iso(created)
                if turn_dt and edt and edt > turn_dt:
                    continue                            # edge didn't exist yet
                fan.add(oi)
                if oi == tt.gr:
                    found.append((srank, si, rel, dlen, created))
                else:
                    noise_rows.append((rel, dlen, deg[oi]))
                    nm = node_meta.get(oi)
                    if nm:
                        noise_tgt_type[nm[0]] += 1
        per_turn.append({'key': key, 'rescued': bool(found),
                         'fanout': len(fan), 'gold_rank': gold_rank,
                         'stratum': v['stratum']})
        gm = node_meta.get(tt.gr, (None, None, 0))
        for (srank, si, rel, dlen, created) in found:
            sm = node_meta.get(si, (None, None, 0))
            edt = iso(created)
            rescues.append({
                'key': key, 'seed_rank': srank, 'rel': rel, 'desc_len': dlen,
                'edge_age_d': (turn_dt - edt).days if (turn_dt and edt) else None,
                'seed_type': sm[0], 'gold_type': gm[0],
                'gold_size': gm[2], 'seed_size': sm[2],
                'gold_deg': deg[tt.gr], 'seed_deg': deg[si],
            })

    n_miss = len(per_turn)
    n_resc = sum(1 for p in per_turn if p['rescued'])
    L = ['# 1-hop anatomy — rescue vs noise character (clean valid misses)', '',
         'misses analyzed: %d · gold within 1 hop of a top-%d seed: %d (%.0f%%)'
         % (n_miss, K_SEEDS, n_resc, 100*n_resc/max(1, n_miss)),
         'mean fan-out per turn: %.0f nodes (the noise a blind +1-hop adds)'
         % np.mean([p['fanout'] for p in per_turn]), '',
         'rescue rate by stratum: ' + ' · '.join(
             '%s %.0f%%' % (st, 100*np.mean([p['rescued'] for p in per_turn
                                             if p['stratum'] == st]))
             for st in ('cue', 'window', 'session')), '']

    # relation lift table
    noise_rel = Counter(rel for rel, _, _ in noise_rows)
    resc_rel = Counter(r['rel'] for r in rescues)
    tot_r, tot_n = sum(resc_rel.values()), len(noise_rows)
    L += ['## Edge-relation character — rescue share vs noise share (lift)', '',
          '| relation | rescue n | rescue % | noise % | LIFT |',
          '|---|---|---|---|---|']
    for rel, c in resc_rel.most_common(14):
        rs = 100*c/tot_r
        ns = 100*noise_rel.get(rel, 0)/max(1, tot_n)
        L.append('| %s | %d | %.1f%% | %.1f%% | %s%.1f× |'
                 % (rel, c, rs, ns, '**' if rs > 2*ns else '', rs/max(ns, 0.05)))
    L.append('')

    # scalar characters
    def med(a):
        a = [x for x in a if x is not None]
        return np.median(a) if a else float('nan')
    L += ['## Scalar character — rescue vs noise medians', '',
          '| axis | rescue | noise |', '|---|---|---|',
          '| edge description length | %.0f | %.0f |'
          % (med([r['desc_len'] for r in rescues]),
             med([dl for _, dl, _ in noise_rows])),
          '| neighbor (target) degree | %.0f | %.0f |'
          % (med([r['gold_deg'] for r in rescues]),
             med([dg for _, _, dg in noise_rows])),
          '| seed rank carrying the hop | %.0f | — |'
          % med([r['seed_rank'] for r in rescues]),
          '| edge age at turn (days) | %.0f | — |'
          % med([r['edge_age_d'] for r in rescues]),
          '| gold (target) content size | %.0f | — |'
          % med([r['gold_size'] for r in rescues]), '']
    sr = Counter(r['seed_rank'] for r in rescues)
    L += ['seed-rank distribution of rescuing hops: '
          + ' · '.join('r%d:%d' % (k, sr[k]) for k in sorted(sr)), '']
    # gold/seed type character
    L += ['## Node-type character', '',
          'rescued-gold types: ' + ', '.join('%s %d' % kv for kv in Counter(
              r['gold_type'] for r in rescues).most_common(8)),
          '',
          'noise-neighbor types (top): ' + ', '.join('%s %d' % kv for kv in
                                                     noise_tgt_type.most_common(8)), '']

    # ── FILTER CURVE: selective walk design ──
    filters = [
        ('ALL edges (blind +1hop)', lambda r: True, lambda rel, dl, dg: True),
        ('drop co_accessed', None, lambda rel, dl, dg: rel != 'co_accessed'),
        ('drop co_accessed+community+related*',
         None, lambda rel, dl, dg: rel not in
         ('co_accessed', 'community_member', 'related_to', 'related')),
        ('semantic only + desc≥80 chars', None,
         lambda rel, dl, dg: rel not in
         ('co_accessed', 'community_member', 'related_to', 'related')
         and dl >= 80),
        ('semantic + desc≥80 + non-hub (deg≤60)', None,
         lambda rel, dl, dg: rel not in
         ('co_accessed', 'community_member', 'related_to', 'related')
         and dl >= 80 and dg <= 60),
    ]
    L += ['## Filter curve — %% rescues kept vs %% fan-out cut', '',
          '| filter | rescues kept | fan-out kept |', '|---|---|---|']
    resc_all = len(rescues)
    noise_all = len(noise_rows)
    for name, _, keep in filters:
        rk = sum(1 for r in rescues if keep(r['rel'], r['desc_len'], r['gold_deg']))
        nk = sum(1 for rel, dl, dg in noise_rows if keep(rel, dl, dg))
        L.append('| %s | %.0f%% (%d/%d) | %.0f%% |'
                 % (name, 100*rk/max(1, resc_all), rk, resc_all,
                    100*nk/max(1, noise_all)))
    L.append('')

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
