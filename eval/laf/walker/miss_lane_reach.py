"""Per-miss reverse diagnosis — which lane could have reached it within 1 hop?

Tom's reverse ask (2026-07-29): for every turn today's LAF misses, say which
lane could have reached the gold organically or with ONE hop; where a hop did
it, report the characteristics of the edge and the lane and what that means.
Then: for golds NO lane touched at all, say what those golds ARE and what makes
them gold to the moment.

DEFINITIONS
  miss        gold rank > 5 under the shipped mix (λ=0.65) — today's LAF output
  organic     gold inside that lane's OWN top-25 (a real spike, not a 0.1
              cosine — Tom's explicit exclusion)
  hop         gold one complementary-verb edge from that lane's top-25, edge
              existing at turn time
  UNTOUCHED   no lane reaches it organically AND no lane reaches it in 1 hop
  exclusive   exactly one lane reaches it (organic or hop) — the arbitration
              population: a router that picked that lane would win the turn

OUTPUTS
  A. per-lane reach table (organic / hop / exclusive) + edge characteristics
     for the hop-reached (verbs, description length, seed rank) + turn/gold
     characteristics (gold age, cue length, gold type)
  B. the UNTOUCHED class: profile vs reached, and a printed sample of cases
     (operator message + gold title/type/age) for a qualitative read

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/miss_lane_reach.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from edge_fusion_census import LAM, iso, verb_class                 # noqa: E402

REPORT = OUT_DIR / 'miss_lane_reach.md'
LANES = ('maxsim', 'sit', 'idf', 'pick', 'enc', 'mh')
SEED_K = 25
N_SAMPLE = 14


def main():
    aspects = json.loads(
        (__import__('pathlib').Path(__file__).resolve().parents[3] /
         'servers/scales/s2/aspects_v1.json').read_text())
    corrective_all = set(
        (aspects.get('correction_improvement') or {}).get('edge_relations') or [])

    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    master = idx['master']
    m2i = {sid: i for i, sid in enumerate(master)}
    n_nodes = idx['n_nodes']
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}

    b = open_brain_ro()
    node_meta = {}
    for nid, typ, created, clen, title in b.execute(
            'SELECT id, type, created_at, LENGTH(content), title FROM nodes'):
        i = m2i.get(nid)
        if i is not None:
            node_meta[i] = (typ, created, clen or 0, title or '')
    adj = defaultdict(list)
    partners = defaultdict(set)
    for src, tgt, rel, dlen, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, "
            "LENGTH(COALESCE(r.description,'')), e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        vc = verb_class(rel, corrective_all)
        adj[si].append((ti, rel, vc, dlen or 0, created))
        adj[ti].append((si, rel, vc, dlen or 0, created))
        partners[si].add(ti)
        partners[ti].add(si)
    b.close()
    deg = {k: len(v) for k, v in partners.items()}

    turns, _enr, n = D.build_corpus('2026-05-11')
    gains = np.array([A.GAINS[ln] for ln in A.LANES])

    per_lane = {ln: defaultdict(int) for ln in LANES}
    hop_edges = {ln: [] for ln in LANES}      # (verb, desc_len, seed_rank)
    lane_gold = {ln: [] for ln in LANES}      # (gold_age_d, cue_len, type)
    untouched, reached = [], []
    n_miss = 0

    for t in turns:
        U = np.flatnonzero(t['alive'])
        if U.size < 20:
            continue
        Z = np.column_stack([t['zl'][ln][U] for ln in A.LANES])
        f0 = Z @ gains
        if not np.isfinite(f0).any() or f0.std() <= 1e-9:
            continue
        zf0 = (f0 - f0.mean()) / f0.std()
        zmh = zn(t['mh'])[U]
        mix = LAM * zf0 + (1.0 - LAM) * zmh
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        order = np.argsort(-fin)
        gpos = np.flatnonzero(U[order] == t['gr'])
        if not gpos.size:
            continue
        gold_rank = int(gpos[0]) + 1
        if gold_rank <= 5:
            continue
        n_miss += 1
        gr = int(t['gr'])
        tdt = t.get('turn_dt')
        bd = bundles.get(t['key']) or {}
        cue = bd.get('op_text') or ''
        gm = node_meta.get(gr, ('?', '', 0, ''))
        gdt = iso(gm[1])
        gold_age = (tdt - gdt).days if (tdt and gdt) else None

        reach = {}                            # lane -> 'organic' | 'hop'
        edge_of = {}
        for ln in LANES:
            arr = np.full(n_nodes, -np.inf)
            arr[U] = t['zl'][ln][U] if ln != 'mh' else zmh
            k = min(SEED_K, U.size)
            sel = U[np.argsort(-arr[U])[:k]]
            seeds = [int(x) for x in sel]
            if gr in seeds:
                reach[ln] = 'organic'
                continue
            for rank, si in enumerate(seeds, 1):
                found = None
                for (oi, rel, vc, dlen, created) in adj.get(si, ()):
                    if oi != gr or vc != 'complementary':
                        continue
                    edt = iso(created)
                    if tdt and edt and edt > tdt:
                        continue
                    found = (rel, dlen, rank, si)
                    break
                if found:
                    reach[ln] = 'hop'
                    edge_of[ln] = found
                    break

        for ln, kind in reach.items():
            per_lane[ln][kind] += 1
            lane_gold[ln].append((gold_age, len(cue), gm[0]))
            if kind == 'hop':
                hop_edges[ln].append(edge_of[ln][:3])
        if len(reach) == 1:
            per_lane[next(iter(reach))]['exclusive'] += 1

        rec = {'key': t['key'], 'rank': gold_rank, 'cue': cue,
               'gold_title': gm[3], 'gold_type': gm[0], 'gold_age': gold_age,
               'gold_size': gm[2], 'gold_deg': deg.get(gr, 0),
               'stratum': t['stratum'], 'nlanes': len(reach),
               'gold_id': master[gr][:8]}
        (reached if reach else untouched).append(rec)

    def prof(rows):
        if not rows:
            return 'n/a'
        ages = sorted(r['gold_age'] for r in rows if r['gold_age'] is not None)
        cues = sorted(len(r['cue']) for r in rows)
        return ('n=%d · gold age med %dd · cue len med %d · deg med %d · '
                'size med %d' % (
                    len(rows), ages[len(ages) // 2] if ages else -1,
                    cues[len(cues) // 2],
                    sorted(r['gold_deg'] for r in rows)[len(rows) // 2],
                    sorted(r['gold_size'] for r in rows)[len(rows) // 2]))

    L = ['# Per-miss reverse diagnosis — which lane reaches within 1 hop', '',
         'misses vs today\'s LAF (gold rank>5, shipped λ=0.65): **%d**' % n_miss,
         '', '## A. Per-lane reach', '',
         '| lane | organic | +1 hop | total | EXCLUSIVE | hop verbs (top) | '
         'hop desc-len med | seed rank med | gold age med | cue len med |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for ln in LANES:
        s = per_lane[ln]
        he = hop_edges[ln]
        verbs = Counter(v for v, _d, _r in he).most_common(3)
        dls = sorted(d for _v, d, _r in he)
        srs = sorted(r for _v, _d, r in he)
        ages = sorted(a for a, _c, _t in lane_gold[ln] if a is not None)
        cues = sorted(c for _a, c, _t in lane_gold[ln])
        L.append('| %s | %d | %d | %d | **%d** | %s | %d | %d | %s | %d |'
                 % (ln, s['organic'], s['hop'], s['organic'] + s['hop'],
                    s['exclusive'],
                    ', '.join('%s(%d)' % v for v in verbs) or '—',
                    dls[len(dls) // 2] if dls else 0,
                    srs[len(srs) // 2] if srs else 0,
                    '%dd' % ages[len(ages) // 2] if ages else '—',
                    cues[len(cues) // 2] if cues else 0))

    L += ['', '## B. The UNTOUCHED class — no lane, no hop', '',
          '- reached by >=1 lane (organic or hop): %s' % prof(reached),
          '- **UNTOUCHED**: %s' % prof(untouched), '',
          '| class | share | types (top 5) | strata |', '|---|---|---|---|']
    for nm, rows in (('reached', reached), ('UNTOUCHED', untouched)):
        L.append('| %s | %.0f%% | %s | %s |'
                 % (nm, 100.0 * len(rows) / max(n_miss, 1),
                    ', '.join('%s(%d)' % x for x in
                              Counter(r['gold_type'] for r in rows).most_common(5)),
                    ', '.join('%s(%d)' % x for x in
                              Counter(r['stratum'] for r in rows).most_common())))

    L += ['', '### UNTOUCHED sample (for qualitative read)', '']
    rng = np.random.default_rng(20260729)
    sample = [untouched[i] for i in
              rng.permutation(len(untouched))[:N_SAMPLE]] if untouched else []
    for r in sample:
        L += ['**cue** (%s, rank %d): `%s`'
              % (r['stratum'], r['rank'], (r['cue'] or '')[:260].replace('\n', ' ')),
              '- gold `%s` [%s, %dd old, deg %d, %d chars]: %s'
              % (r['gold_id'], r['gold_type'], r['gold_age'] or -1,
                 r['gold_deg'], r['gold_size'], r['gold_title'][:130]), '']
    REPORT.write_text('\n'.join(L) + '\n')

    # LEAK-FREE dump for the trigger-register experiment: node payload ONLY,
    # never the cue that missed it. A rewrite written while looking at the cue
    # is hand-fitted to the test; the situation must generalise, not target.
    b2 = open_brain_ro()
    want = {r['gold_id'] for r in untouched}
    out = []
    for nid, typ, title, content in b2.execute(
            'SELECT id, type, title, content FROM nodes'):
        if nid[:8] not in want:
            continue
        kv = dict(b2.execute(
            'SELECT key, value FROM node_metadata_kv WHERE node_id=?',
            (nid,)).fetchall())
        out.append({'id': nid[:8], 'type': typ, 'title': title,
                    'content': (content or '')[:1800],
                    'situation': kv.get('situation', ''),
                    'question': kv.get('question', '')})
    b2.close()
    (OUT_DIR / 'untouched_golds.json').write_text(json.dumps(out, indent=1))
    print('wrote untouched_golds.json (%d nodes, NO cues — leak-free)'
          % len(out))
    print('\n'.join(L[:30]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
