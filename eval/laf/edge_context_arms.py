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
turn-date >= cutoff, gold node still live.

TWO LIMITS ON WHAT THIS CAN CONCLUDE, both deliberate:

1. Time-honesty covers NODES, not EDGES. A cue at ts ranks only nodes created
   at or before ts, but each node's text is built from the graph as it stands
   TODAY — so a node can carry descriptions of edges written after the cue.
   This matters most to the high-top_k arms (only nodes with >5 described edges
   can differ between k=5 and k=15 at all), and it leaks in their favour, so a
   null here is the conservative direction: the true k=15 effect under
   edge-time-honest replay is at most what this measures.

2. Every arm is rebuilt from scratch, so freshness is pinned at 100% for all of
   them. v33 did TWO things — it changed this producer policy AND it fixed the
   invalidation that had left ~44% of edge_context vectors embedding an
   outdated graph snapshot. These arms isolate the policy half. The staleness
   half is held constant and is NOT measured here; do not read a null from this
   probe as "the v33 deploy did nothing".

Metric: reach@k, reported TWO ways per arm, because they answer different
questions and can disagree:

  in-stack  nanmax over all six MaxSim views, as production scores. A view only
            moves a node here when it is the argmax, so this is MAXIMAL dilution
            for a one-view text change — not, as an earlier version of this
            docstring claimed, an undiluted view of it.
  SOLO      the edge_context view scored alone: the undiluted dose-response on
            the text itself. maxsim_decomp.md measured this view alone at 9%
            need@5 against 14% for shipped nanmax and 16% for sum(z), so the
            aggregator demonstrably discards signal this view carries.

If a text change moves SOLO and not in-stack, the finding is about the
AGGREGATOR, not the producer policy — and those imply opposite actions.

Each contrast reports a paired bootstrap CI under common random numbers AND the
McNemar discordant counts b/c, because only discordant cues carry information:
arms that disagree on nine of 792 cues are a nine-trial comparison, and
quoting n=792 hides that.

Run (isolated copy of production; never writes the live brain):
    ./dev python3 eval/laf/edge_context_arms.py [--cutoff 2026-05-11] [--boot 2000]
"""
import argparse
import json
import os
import sqlite3
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
from operators import (MAXSIM_GROUPS, build_field_matrices, query_vec,  # noqa: E402
                       unit)

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
    failed = []
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
            v = unit(blob)          # operators.unit IS production's _unit — never re-roll it
            if v is None:
                failed.append(t)    # do NOT cache a failure: it would replicate into every
                continue            # later arm and read as a consistent, invisible shortfall
            _VEC_CACHE[t] = np.asarray(v, dtype=np.float32)
        if (i // chunk) % 8 == 0:
            print('      embed %d/%d' % (min(i + chunk, len(need)), len(need)), flush=True)
    if failed:
        raise RuntimeError('embedder returned %d unusable blobs (e.g. %r) — refusing to score '
                           'an arm with silently missing vectors' % (len(failed), failed[0][:60]))
    placed = 0
    for nid in ids:
        v = _VEC_CACHE.get(texts_by_node[nid])
        if v is not None:
            M[idx[nid]] = v
            placed += 1
    return M, placed, len(need)


def precompute_base(cues, qvecs, mats, base_groups, created, n_rows):
    """The arm-invariant half of the score, computed once.

    The five base views and the per-cue time mask do not change between arms;
    recomputing them per (arm, cue) cost ~6x the matrix traffic and tens of
    millions of object-dtype string compares per run.
    Returns (base [n_cues x n_rows] float32, mask [n_cues x n_rows] bool).
    """
    base = np.empty((len(cues), n_rows), dtype=np.float32)
    mask = np.empty((len(cues), n_rows), dtype=bool)
    for i, c in enumerate(cues):
        b = np.nanmax(np.stack([mats[g] @ qvecs[c['key']] for g in base_groups]), axis=0)
        base[i] = np.where(np.isnan(b), -np.inf, b)
        mask[i] = created <= c['ts'][:19]          # time-honest: no future NODES
    return base, mask


def score_arm(cues, qvecs, base, mask, ec_mat, gold_row, ec_only=False):
    """Per-cue midpoint rank of the gold. `ec_mat=None` is the no-lane control.

    Midpoint rather than optimistic (`1 + count(>)`): exact ties split the
    position instead of handing the gold the best of them. Measured zero ties
    on this corpus, but both endpoints of an edge receive the SAME description
    string, so a future corpus with duplicate edge text could otherwise let the
    arm with more duplicates quietly flatter itself.
    """
    ranks = np.empty(len(cues), dtype=float)
    buried = 0
    for i, c in enumerate(cues):
        s = base[i]
        if ec_mat is not None:
            e = np.where(np.isnan(ec_mat @ qvecs[c['key']]), -np.inf, ec_mat @ qvecs[c['key']])
            # ec_only: the view SCORED ALONE. nanmax over six views is maximal
            # dilution for a one-view text change — this repo already measured
            # edge_context alone at 9% need@5 vs 14% in-stack (maxsim_decomp.md).
            # If a text change moves this and not the stack, the finding is about
            # the aggregator, not the producer policy.
            s = e if ec_only else np.maximum(s, e)
        s = np.where(mask[i], s, -np.inf)
        gr = gold_row[c['gold']]
        gs = s[gr]
        if not mask[i][gr]:
            # Masked by time/archive = instrument fault (the rank is meaningless).
            # A -inf from "this view has no vector for the gold" is NOT a fault:
            # under ec_only that is a legitimate miss, and 14% of golds have no
            # edge_context vector at all.
            buried += 1
        ranks[i] = 1 + np.sum(s > gs) + 0.5 * (np.sum(s == gs) - 1)
    if buried:
        raise RuntimeError('%d cue(s) have a gold masked out of their own candidate set by the '
                           'time/archive mask — their ranks would be silent permanent misses '
                           'that deflate every arm equally' % buried)
    return ranks


def self_test(cues, qvecs, base, mask, gold_row, n_rows, dim):
    """Positive control: can the edge_context view move the metric AT ALL?

    A genuinely inert lane and a probe that silently drops the lane produce the
    same null, so the null means nothing unless this passes. Per cue, plant the
    cue's own query vector in the gold's edge_context row (cosine 1.0 — the
    strongest signal any text could produce) and re-rank. If the view reaches
    the MaxSim stack, reach@1 must go to ~100%.
    """
    M = np.full((n_rows, dim), np.nan, dtype=np.float32)
    ranks = np.empty(len(cues), dtype=float)
    for i, c in enumerate(cues):
        gr = gold_row[c['gold']]
        M[gr] = qvecs[c['key']]                  # plant, score, unplant
        ranks[i] = score_arm([c], qvecs, base[i:i + 1], mask[i:i + 1], M, gold_row)[0]
        M[gr] = np.nan
    return ranks


def reach_at(ranks, k):
    return 100.0 * float(np.mean(ranks <= k))


def mcnemar(a, b, k):
    """Discordant counts for a paired binary contrast at k, plus the exact p.

    Only discordant cues carry information: a 792-cue corpus whose arms disagree
    on nine of them is a nine-trial comparison, and reporting `n=792` hides that.
    b = arm hits where reference misses, c = the reverse.
    """
    ha, hb = a <= k, b <= k
    nb, nc = int(np.sum(ha & ~hb)), int(np.sum(~ha & hb))
    n = nb + nc
    if n == 0:
        return nb, nc, 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(nb, nc) + 1)) / (2 ** n)
    return nb, nc, min(1.0, 2 * tail)


def paired_bootstrap(a, b, k, draws):
    """CI on reach@k difference (arm - reference) under COMMON random numbers.

    `draws` is one shared [n_boot x n] index matrix, so every arm is resampled
    on the same cue draws and the arms are comparable to each other — not just
    each to the reference. A per-arm rng stream would make a CI depend on how
    many arms happened to be scored before it.
    """
    hit_a, hit_b = (a <= k).astype(float), (b <= k).astype(float)
    diffs = 100.0 * (hit_a[draws].mean(axis=1) - hit_b[draws].mean(axis=1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return 100.0 * (hit_a.mean() - hit_b.mean()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', default='2026-05-11')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--self-test', action='store_true',
                    help='positive control only: prove the edge_context view can move the metric')
    ap.add_argument('--topk-sweep', default='', help='extra top_k values on the noise set, e.g. 3,10,25,40')
    args = ap.parse_args()

    # A non-positive top_k is not a smaller arm: SQLite reads a negative LIMIT
    # as NO limit, so the arm would silently embed every described edge.
    sweep = [int(x) for x in args.topk_sweep.split(',') if x.strip()]
    bad = [k for k in sweep if k < 1]
    if bad:
        sys.exit('--topk-sweep needs positive ints; got %r' % (bad,))

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
        if not all(ca_map.get(nid) for nid in master):
            sys.exit('a node has no created_at — it would read as infinitely old and leak into '
                     'every cue past the time-honest mask')
        # get_all_vectors excludes archived nodes, so master is already the live
        # set. Assert it rather than re-masking: a silent mask would hide the day
        # that default changes, and every node here competes for rank.
        arch = dict(brain.conn.execute('SELECT id, archived FROM nodes').fetchall())
        assert not any(arch.get(nid) for nid in master), 'archived nodes reached the candidate set'
        print('field matrices: %d nodes x %d dim over %s' % (n_rows, dim, base_groups))

        cues = [c for c in cues if c['gold'] in idx]
        print('cues with a live, embedded gold: %d' % len(cues))
        qvecs = {}
        for c in cues:
            qv = query_vec(c['q'])
            if qv is not None:
                qvecs[c['key']] = qv
        cues = [c for c in cues if c['key'] in qvecs]
        print('cues with a query vector: %d' % len(cues))
        if not cues:
            sys.exit('no scorable cues — refusing to write a report of nan')
        gold_row = {c['gold']: idx[c['gold']] for c in cues}

        base, mask = precompute_base(cues, qvecs, mats, base_groups, created, n_rows)

        ctrl = self_test(cues, qvecs, base, mask, gold_row, n_rows, dim)
        print('SELF-TEST (planted gold vector): reach@1=%.2f%% reach@5=%.2f%% median_rank=%.0f'
              % (reach_at(ctrl, 1), reach_at(ctrl, 5), np.median(ctrl)), flush=True)
        if reach_at(ctrl, 1) < 99.0:
            raise RuntimeError('POSITIVE CONTROL FAILED: planting a perfect edge_context vector '
                               'left reach@1 at %.2f%% — the view is not reaching the score, so a '
                               'null result from this probe means nothing.' % reach_at(ctrl, 1))
        if args.self_test:
            return

        arms = [('B_old  (k=5,  cm-only)', 5, OLD_EXCLUDED),
                ('A_new  (k=15, noise)', 15, noise),
                ('C      (k=5,  noise)', 5, noise),
                ('D      (k=15, cm-only)', 15, OLD_EXCLUDED)]
        arms += [('S      (k=%d, noise)' % k, k, noise) for k in sweep]
        names = [a[0] for a in arms]
        if len(set(names)) != len(names):
            sys.exit('duplicate arm names %r — their ranks would overwrite each other' % (names,))

        results, ranks_by_arm = [], {}
        orig_seam = brain._graph._edge_context_excluded_fn
        try:
            for name, top_k, excl in arms:
                t0 = time.time()
                texts = arm_texts(brain, master, top_k, excl)
                M, placed, fresh = embed_matrix(texts, idx, n_rows, dim)
                ranks = score_arm(cues, qvecs, base, mask, M, gold_row)
                solo = score_arm(cues, qvecs, base, mask, M, gold_row, ec_only=True)
                ranks_by_arm[name] = ranks
                ranks_by_arm['SOLO ' + name] = solo
                chars = int(np.mean([len(t) for t in texts.values()])) if texts else 0
                results.append({'arm': name, 'top_k': top_k, 'n_excluded': len(excl),
                                'eligible_nodes': len(texts), 'vectors': placed,
                                'mean_chars': chars,
                                'reach': {k: reach_at(ranks, k) for k in KS},
                                'solo_reach': {k: reach_at(solo, k) for k in KS},
                                'median_rank': float(np.median(ranks)),
                                'secs': round(time.time() - t0, 1)})
                print('  %-24s eligible=%-6d chars=%-4d  in-stack r@5=%5.2f%%  SOLO r@5=%5.2f%%  (%.0fs)'
                      % (name, len(texts), chars, reach_at(ranks, 5), reach_at(solo, 5),
                         time.time() - t0), flush=True)
        finally:
            brain._graph._edge_context_excluded_fn = orig_seam

        # no-lane control: what recall looks like with edge_context absent entirely
        CTRL = 'Z_no_lane (control)'
        ranks_by_arm[CTRL] = score_arm(cues, qvecs, base, mask, None, gold_row)
        results.append({'arm': CTRL, 'top_k': None, 'n_excluded': None,
                        'eligible_nodes': 0, 'vectors': 0, 'mean_chars': 0,
                        'reach': {k: reach_at(ranks_by_arm[CTRL], k) for k in KS},
                        'median_rank': float(np.median(ranks_by_arm[CTRL])), 'secs': 0})

        REF = 'B_old  (k=5,  cm-only)'
        rng = np.random.default_rng(20260913)
        draws = rng.integers(0, len(cues), (args.boot, len(cues)))   # common random numbers
        for r in results:
            r['delta'] = {}
            for k in KS:
                d, lo, hi = paired_bootstrap(ranks_by_arm[r['arm']], ranks_by_arm[REF], k, draws)
                nb, nc, pv = mcnemar(ranks_by_arm[r['arm']], ranks_by_arm[REF], k)
                r['delta'][k] = {'d': d, 'lo': lo, 'hi': hi, 'b': nb, 'c': nc, 'p': pv}
            r['delta5'] = r['delta'][5]

        out = {'ranks': {k: [float(x) for x in v] for k, v in ranks_by_arm.items()},
               'cue_keys': [c['key'] for c in cues],
               'revision': rev, 'model': model, 'cutoff': args.cutoff,
               'sqlite': sqlite3.sqlite_version, 'self_test_reach1': reach_at(ctrl, 1),
               'n_cues': len(cues), 'n_nodes': n_rows, 'live_config': live_cfg,
               'noise_aspect': sorted(noise), 'base_groups': base_groups, 'arms': results}
        write_report(out)
        print('\nreport -> %s' % REPORT)


def write_report(out):
    L = ['# edge_context producer arms', '',
         '- revision `%s` · model `%s` · sqlite `%s` · positive control reach@1 %.1f%%'
         % (out['revision'], out['model'], out['sqlite'], out['self_test_reach1']),
         '- corpus corpus_v2 valid, cutoff %s → **%d cues**, %d embedded nodes'
         % (out['cutoff'], out['n_cues'], out['n_nodes']),
         '- live `edge_context` config on the copy: `%r`' % (out['live_config'],),
         '- noise aspect (%d): %s' % (len(out['noise_aspect']), ', '.join(out['noise_aspect'])),
         '- MaxSim base views: %s (+ edge_context per arm)' % ', '.join(out['base_groups']), '',
         '| arm | top_k | excl | eligible | chars | r@1 | r@5 | r@10 | r@25 | SOLO r@5 | Δr@5 vs OLD [95% CI] | b/c | McNemar p |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in out['arms']:
        d = r['delta5']
        solo = r.get('solo_reach', {}).get(5)
        L.append('| %s | %s | %s | %d | %d | %.2f | %.2f | %.2f | %.2f | %s | %+.2f [%+.2f, %+.2f] | %d/%d | %.3f |'
                 % (r['arm'], r['top_k'] if r['top_k'] is not None else '—',
                    r['n_excluded'] if r['n_excluded'] is not None else '—',
                    r['eligible_nodes'], r['mean_chars'],
                    r['reach'][1], r['reach'][5], r['reach'][10], r['reach'][25],
                    ('%.2f' % solo) if solo is not None else '—',
                    d['d'], d['lo'], d['hi'], d['b'], d['c'], d['p']))
    with open(REPORT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    with open(REPORT.replace('.md', '.json'), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
