"""Direction read-out — does STORED edge direction carry gold-relevant info?

Tom's override thesis (2026-07-29): if verbs are pre-classified by algebra
(asymmetric/symmetric/...), the stored direction column is derivable
(verb + node age, the 96% new→old convention) and traversal can override it.
Empirical test on the hop-anatomy population (clean valid misses, top-5 mix
seeds, time-honest edges): for every RESCUE hop (seed—gold edge) and every
NOISE hop, was the neighbor reached ALONG stored direction (seed=source) or
AGAINST it (seed=target)?

Readouts:
  - rescues via out / in, per verb — if rescues split both ways, any
    direction-honoring walk LOSES golds → direction is not a traversal
    constraint (override confirmed);
  - the same split for noise → what fan-out a direction-honoring walk would
    cut, i.e. what direction could buy IF rescues were one-way;
  - grouped by the research algebra classes (86714339) — is the class
    load-bearing for traversal or just taxonomy?

Read-only. Run:  ./dev python3 eval/laf/walker/role_direction_readout.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn                                         # noqa: E402

CUTOFF = '2026-05-11'
LAM = 0.65
K_SEEDS = 5
REPORT = OUT_DIR / 'role_direction_readout.md'

# Algebra classes per the directed-vs-symmetric research synthesis
# (id:86714339). SYMMETRIC verbs: stored direction is creation-accident by
# definition (and co_accessed contamination, id:63321aff). Everything else
# is treated as directed-candidate and judged by the data.
SYMMETRIC = {'co_accessed', 'co_anchored', 'similar_to', 'related',
             'related_to', 'parallels', 'community_member'}


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    b = open_brain_ro()
    # DIRECTED adjacency: node_i -> [(other_i, rel, created, out?)]
    adj = defaultdict(list)
    for src, tgt, rel, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        adj[si].append((ti, rel, created, True))    # out: si=source
        adj[ti].append((si, rel, created, False))   # in:  ti=target
    b.close()

    resc = defaultdict(lambda: [0, 0])    # verb -> [via_out, via_in]
    noise = defaultdict(lambda: [0, 0])
    n_miss = n_resc_turn = 0
    resc_turn_out_only = resc_turn_in_only = resc_turn_both_dirs = 0
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
        if int(np.where(order == tt.gr)[0][0]) + 1 <= 5:
            continue
        n_miss += 1
        turn_dt = iso(bd['ts'])
        hit_dirs = set()
        for si in (int(x) for x in order[:K_SEEDS]):
            for (oi, rel, created, is_out) in adj.get(si, ()):
                edt = iso(created)
                if turn_dt and edt and edt > turn_dt:
                    continue
                slot = 0 if is_out else 1
                if oi == tt.gr:
                    resc[rel][slot] += 1
                    hit_dirs.add(is_out)
                else:
                    noise[rel][slot] += 1
        if hit_dirs:
            n_resc_turn += 1
            if hit_dirs == {True}:
                resc_turn_out_only += 1
            elif hit_dirs == {False}:
                resc_turn_in_only += 1
            else:
                resc_turn_both_dirs += 1

    def tot(d):
        o = sum(v[0] for v in d.values())
        i = sum(v[1] for v in d.values())
        return o, i

    L = ['# Direction read-out — stored direction vs gold rescue', '',
         'misses: %d · rescued turns: %d — reached via OUT-edge only: %d, '
         'IN-edge only: %d, both: %d' % (n_miss, n_resc_turn,
                                         resc_turn_out_only,
                                         resc_turn_in_only,
                                         resc_turn_both_dirs), '']
    for name, keep in (('DIRECTED-CANDIDATE verbs',
                        lambda r: r not in SYMMETRIC),
                       ('SYMMETRIC verbs (direction = accident by design)',
                        lambda r: r in SYMMETRIC)):
        ro = {r: v for r, v in resc.items() if keep(r)}
        no = {r: v for r, v in noise.items() if keep(r)}
        ro_o, ro_i = tot(ro)
        no_o, no_i = tot(no)
        L += ['## %s' % name, '',
              'rescue hops: out %d / in %d · noise hops: out %d / in %d'
              % (ro_o, ro_i, no_o, no_i), '',
              '| verb | rescue out | rescue in | noise out | noise in |',
              '|---|---|---|---|---|']
        for r, (o, i) in sorted(ro.items(), key=lambda x: -(x[1][0] + x[1][1])):
            nn = no.get(r, [0, 0])
            L.append('| %s | %d | %d | %d | %d |' % (r, o, i, nn[0], nn[1]))
        L.append('')
    # the walk-policy table: what each direction policy keeps
    ro_o, ro_i = tot({r: v for r, v in resc.items() if r not in SYMMETRIC})
    no_o, no_i = tot({r: v for r, v in noise.items() if r not in SYMMETRIC})
    sym_r = tot({r: v for r, v in resc.items() if r in SYMMETRIC})
    sym_n = tot({r: v for r, v in noise.items() if r in SYMMETRIC})
    L += ['## Walk-policy table (directed verbs honor policy; symmetric always both ways)', '',
          '| policy | rescue hops kept | noise hops kept |', '|---|---|---|']
    for pname, rk, nk in (
            ('both ways (direction ignored)',
             ro_o + ro_i + sum(sym_r), no_o + no_i + sum(sym_n)),
            ('out only (honor stored direction)',
             ro_o + sum(sym_r), no_o + sum(sym_n)),
            ('in only (reverse)', ro_i + sum(sym_r), no_i + sum(sym_n))):
        base_r = ro_o + ro_i + sum(sym_r)
        base_n = no_o + no_i + sum(sym_n)
        L.append('| %s | %d (%.0f%%) | %d (%.0f%%) |'
                 % (pname, rk, 100.0 * rk / max(base_r, 1),
                    nk, 100.0 * nk / max(base_n, 1)))
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
