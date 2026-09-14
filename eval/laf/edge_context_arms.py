#!/usr/bin/env python3
"""edge_context producer-policy arms — did the v33 text change help recall?

Two knobs shipped together (main 7773993, merged d589321, deployed 2026-09-13):
    top_k      5 -> 15                  (now the `edge_context` interaction config)
    excluded   {community_member} -> the whole noise aspect (10 relations)

This runs the 2x2 so the two can be told apart. Both knobs are injected at
their PRODUCTION seams — `limit` is the producer's own parameter, and
GraphDAL._edge_context_excluded_fn is the zero-arg callable Brain installs
(brain.py:352, "a callable, never a set copied in"). So every arm runs the
shipped producer verbatim: no patched tree, no alternate aspects JSON, no
reimplemented SQL. The text composition mirrors the one consumer
(brain_recall.backfill_vectors): '. '.join(desc[:EMBEDDING_FIELD_CHAR_LIMIT]).

Corpus: walker/corpus_v2_{bundles,verdicts}.jsonl — verdict == 'valid',
turn-date >= cutoff, gold node still live. Time-honest: a cue at ts ranks only
nodes created at or before ts.

Metric: MaxSim-lane reach@k. edge_context enters LAF only through the MaxSim
lane (recall_laf.MAXSIM_VIEWS), so this is where the change is observable and
undiluted. Paired bootstrap CI against the OLD arm.

Run (isolated copy of production; never writes the live brain):
    ./dev python3 eval/laf/edge_context_arms.py [--cutoff 2026-05-11] [--boot 2000]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from tests.isolated_brain import IsolatedBrain                        # noqa: E402
from servers import embedder                                          # noqa: E402
from servers.pipeline_contract import EMBEDDING_FIELD_CHAR_LIMIT      # noqa: E402
from operators import MAXSIM_GROUPS, build_field_matrices, query_vec  # noqa: E402

WALKER = os.path.join(HERE, 'walker')
BUNDLES = os.path.join(WALKER, 'corpus_v2_bundles.jsonl')
VERDICTS = os.path.join(WALKER, 'corpus_v2_verdicts.jsonl')
REPORT = os.path.join(HERE, 'edge_context_arms.md')
EC = 'edge_context'
OLD_EXCLUDED = frozenset({'community_member'})
KS = (1, 5, 10, 25)


def load_corpus(cutoff):
    verdict = {}
    with open(VERDICTS) as fh:
        for line in fh:
            r = json.loads(line)
            verdict[r['key']] = r.get('verdict')
    cues, dropped = [], 0
    with open(BUNDLES) as fh:
        for line in fh:
            b = json.loads(line)
            if verdict.get(b['key']) != 'valid':
                continue
            ts = b.get('ts') or ''
            gold = (b.get('gold') or {}).get('id')
            q = (b.get('op_text') or '').strip()
            if not gold or not q or ts[:10] < cutoff:
                dropped += 1
                continue
            cues.append({'key': b['key'], 'q': q, 'gold': gold, 'ts': ts,
                         'stratum': b.get('v0_stratum') or '?'})
    return cues, dropped


def arm_texts(brain, node_ids, top_k, excluded):
    """Production producer, both knobs at their real seams."""
    brain._graph._edge_context_excluded_fn = lambda: frozenset(excluded)
    assert brain._graph.edge_context_excluded == frozenset(excluded), 'exclusion seam did not take'
    out = {}
    for nid in node_ids:
        parts = [d[:EMBEDDING_FIELD_CHAR_LIMIT]
                 for d in brain._graph.get_edge_descriptions_for(nid, limit=top_k)]
        if parts:
            out[nid] = '. '.join(parts)
    return out


_VEC_CACHE = {}          # text -> unit vector; arms share it (59% of texts repeat)


def embed_matrix(texts_by_node, idx, n_rows, dim, chunk=256):
    """[N x dim] with NaN rows where the node has no edge_context text.

    Chunked and cached on purpose. Handing ~9.3k of these texts to
    embed_batch in one call peaks at ~15GB and runs at ~32 texts/s (they
    average ~780 chars, not the ~100 a naive throughput test suggests);
    chunking caps resident memory, and the cache skips the 59% of texts that
    are byte-identical across arms (a node with <=5 described edges produces
    the same string at top_k 5 and 15).
    """
    M = np.full((n_rows, dim), np.nan, dtype=np.float32)
    ids = [nid for nid in texts_by_node if nid in idx]
    need, seen = [], set()
    for nid in ids:
        t = texts_by_node[nid]
        if t not in _VEC_CACHE and t not in seen:
            seen.add(t)
            need.append(t)
    for i in range(0, len(need), chunk):
        part = need[i:i + chunk]
        blobs = embedder.embed_batch(part, kind='document')
        if len(blobs) != len(part):
            raise RuntimeError('embedder returned %d blobs for %d texts — model not ready?'
                               % (len(blobs), len(part)))
        for t, blob in zip(part, blobs):
            v = None
            if blob:
                a = np.frombuffer(blob, dtype=np.float32)
                nrm = np.linalg.norm(a)
                if nrm > 0:
                    v = (a / nrm).astype(np.float32)
            _VEC_CACHE[t] = v
        if (i // chunk) % 8 == 0:
            print('      embed %d/%d' % (min(i + chunk, len(need)), len(need)), flush=True)
    placed = 0
    for nid in ids:
        v = _VEC_CACHE.get(texts_by_node[nid])
        if v is not None:
            M[idx[nid]] = v
            placed += 1
    return M, placed, len(need)


def score_arm(cues, qvecs, mats, groups, gold_row, created, live_mask):
    """Per-cue rank of the gold under time-honest MaxSim. Returns list of ranks."""
    stack_groups = [g for g in groups if g in mats]
    ranks = []
    for c in cues:
        qv = qvecs[c['key']]
        s = np.nanmax(np.stack([mats[g] @ qv for g in stack_groups]), axis=0)
        s = np.where(np.isnan(s), -np.inf, s)
        s = np.where(created <= c['ts'][:19], s, -np.inf)   # time-honest
        s = np.where(live_mask, s, -np.inf)                # archived aren't candidates
        gr = gold_row[c['gold']]
        ranks.append(int(1 + np.sum(s > s[gr])))
    return np.array(ranks)


def reach_at(ranks, k):
    return 100.0 * float(np.mean(ranks <= k))


def paired_bootstrap(a, b, k, n_boot, rng):
    """CI on reach@k difference (arm - reference), paired over cues."""
    n = len(a)
    hit_a, hit_b = (a <= k).astype(float), (b <= k).astype(float)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = 100.0 * (hit_a[s].mean() - hit_b[s].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return 100.0 * (hit_a.mean() - hit_b.mean()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', default='2026-05-11')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--topk-sweep', default='', help='extra top_k values on the noise set, e.g. 3,10,25,40')
    args = ap.parse_args()

    rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    cues, dropped = load_corpus(args.cutoff)
    print('corpus: %d valid cues (cutoff %s, %d dropped)' % (len(cues), args.cutoff, dropped))

    with IsolatedBrain() as env:
        brain = env.brain
        brain.recall(query='warm', limit=1)          # warms the embedder via production path
        model = embedder.stats.get('model_name') or ''
        assert embedder.is_ready(), 'embedder not ready'
        assert EC in MAXSIM_GROUPS, 'edge_context missing from MAXSIM_GROUPS: %r' % (MAXSIM_GROUPS,)

        noise = frozenset(brain.aspects.structural_exclusions)
        live_cfg = brain.get_interaction_config('edge_context')
        print('noise aspect (%d): %s' % (len(noise), ', '.join(sorted(noise))))
        print('live edge_context config: %r' % (live_cfg,))

        base_groups = [g for g in MAXSIM_GROUPS if g != EC]
        master, idx, mats = build_field_matrices(brain, model, base_groups)
        n_rows = len(master)
        dim = mats[base_groups[0]].shape[1]
        # created_at carries two ISO spellings ('...+00:00' and '...Z'); their
        # first 19 chars are identical in shape, so compare on that prefix.
        ca_map = dict(brain.conn.execute('SELECT id, created_at FROM nodes').fetchall())
        created = np.array([(ca_map.get(nid) or '')[:19] for nid in master], dtype=object)
        # Archived nodes are not recall candidates and are not in the backfill's
        # candidate set, so they neither get text nor compete for rank.
        arch = dict(brain.conn.execute('SELECT id, archived FROM nodes').fetchall())
        live_mask = np.array([not arch.get(nid) for nid in master], dtype=bool)
        live_ids = [nid for nid in master if not arch.get(nid)]
        print('field matrices: %d nodes x %d dim over %s' % (n_rows, dim, base_groups))

        cues = [c for c in cues if c['gold'] in idx and not arch.get(c['gold'])]
        print('cues with a live, embedded gold: %d' % len(cues))
        gold_row = {c['gold']: idx[c['gold']] for c in cues}

        qvecs = {}
        for c in cues:
            qv = query_vec(c['q'])
            if qv is not None:
                qvecs[c['key']] = qv
        cues = [c for c in cues if c['key'] in qvecs]
        print('cues with a query vector: %d' % len(cues))

        arms = [('B_old  (k=5,  cm-only)', 5, OLD_EXCLUDED),
                ('A_new  (k=15, noise)', 15, noise),
                ('C      (k=5,  noise)', 5, noise),
                ('D      (k=15, cm-only)', 15, OLD_EXCLUDED)]
        for k in [int(x) for x in args.topk_sweep.split(',') if x.strip()]:
            arms.append(('S      (k=%-2d noise)' % k, k, noise))

        results, ranks_by_arm = [], {}
        for name, top_k, excl in arms:
            t0 = time.time()
            texts = arm_texts(brain, live_ids, top_k, excl)
            M, placed, fresh = embed_matrix(texts, idx, n_rows, dim)
            mats[EC] = M
            ranks = score_arm(cues, qvecs, mats, base_groups + [EC], gold_row, created, live_mask)
            ranks_by_arm[name] = ranks
            chars = int(np.mean([len(t) for t in texts.values()])) if texts else 0
            results.append({'arm': name, 'top_k': top_k, 'n_excluded': len(excl),
                            'eligible_nodes': len(texts), 'vectors': placed,
                            'mean_chars': chars,
                            'reach': {k: reach_at(ranks, k) for k in KS},
                            'median_rank': float(np.median(ranks)),
                            'secs': round(time.time() - t0, 1)})
            print('  %-24s eligible=%-6d vecs=%-6d new_embeds=%-6d chars=%-4d reach@5=%5.2f%%  (%.0fs)'
                  % (name, len(texts), placed, fresh, chars, reach_at(ranks, 5), time.time() - t0), flush=True)

        # no-lane control: what recall looks like with edge_context absent entirely
        mats.pop(EC, None)
        ranks_none = score_arm(cues, qvecs, mats, base_groups, gold_row, created, live_mask)
        ranks_by_arm['Z_no_lane'] = ranks_none
        results.append({'arm': 'Z_no_lane (control)', 'top_k': None, 'n_excluded': None,
                        'eligible_nodes': 0, 'vectors': 0, 'mean_chars': 0,
                        'reach': {k: reach_at(ranks_none, k) for k in KS},
                        'median_rank': float(np.median(ranks_none)), 'secs': 0})

        ref = ranks_by_arm['B_old  (k=5,  cm-only)']
        rng = np.random.default_rng(20260913)
        for r in results:
            a = ranks_by_arm[r['arm'] if r['arm'] in ranks_by_arm else 'Z_no_lane']
            d, lo, hi = paired_bootstrap(a, ref, 5, args.boot, rng)
            r['delta5'] = {'d': d, 'lo': lo, 'hi': hi}

        out = {'ranks': {k: [int(x) for x in v] for k, v in ranks_by_arm.items()},
               'cue_keys': [c['key'] for c in cues],
               'revision': rev, 'model': model, 'cutoff': args.cutoff,
               'n_cues': len(cues), 'n_nodes': n_rows, 'live_config': live_cfg,
               'noise_aspect': sorted(noise), 'base_groups': base_groups,
               'db_dir': env.db_dir, 'arms': results}
        write_report(out)
        print('\nreport -> %s' % REPORT)


def write_report(out):
    L = ['# edge_context producer arms', '',
         '- revision `%s` · model `%s` · isolated copy `%s`' % (out['revision'], out['model'], out['db_dir']),
         '- corpus corpus_v2 valid, cutoff %s → **%d cues**, %d embedded nodes'
         % (out['cutoff'], out['n_cues'], out['n_nodes']),
         '- live `edge_context` config on the copy: `%r`' % (out['live_config'],),
         '- noise aspect (%d): %s' % (len(out['noise_aspect']), ', '.join(out['noise_aspect'])),
         '- MaxSim base views: %s (+ edge_context per arm)' % ', '.join(out['base_groups']), '',
         '| arm | top_k | excl | eligible | vectors | chars | r@1 | r@5 | r@10 | r@25 | med rank | Δr@5 vs OLD [95% CI] |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in out['arms']:
        d = r['delta5']
        L.append('| %s | %s | %s | %d | %d | %d | %.2f | %.2f | %.2f | %.2f | %.0f | %+.2f [%+.2f, %+.2f] |'
                 % (r['arm'], r['top_k'] if r['top_k'] else '—',
                    r['n_excluded'] if r['n_excluded'] is not None else '—',
                    r['eligible_nodes'], r['vectors'], r['mean_chars'],
                    r['reach'][1], r['reach'][5], r['reach'][10], r['reach'][25],
                    r['median_rank'], d['d'], d['lo'], d['hi']))
    with open(REPORT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    with open(REPORT.replace('.md', '.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
