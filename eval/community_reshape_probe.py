#!/usr/bin/env python3
"""Reshape-diff feasibility probe — can structure be re-derived every run?

Tests the architecture question behind the S2 community plasticity work:
if the decoder re-derived the WHOLE graph's cluster structure every run
(algorithmic, free) and the encoder judged only the structural diff
(splits / merges / births / dispersals), would identity survive and what
would the encoder's bill be?

Today the decoder clusters ONLY unplaced nodes — placed nodes exit the
clustering population forever, so placed structure is frozen by
construction. This probe runs the decoder's own steps 1-4b (typed
adjacency -> z-score seeding -> validation -> subset absorption) with no
stored-community anchoring, producing the fresh partition, then measures:

  A  what the fresh partition looks like (sizes, count)
  B  identity survival — % of stored communities with a clear fresh
     counterpart (Jaccard at 0.3 / 0.5 / 0.7), vs split / dispersed
  C  mass decomposition — do the >40-member giants break into real
     sub-clusters when re-derived freely?
  D  weekly structural velocity — diff of fresh partitions on the graph
     as-of one week ago vs now (recent rows archived on the throwaway
     copy); events/week is the encoder-cost number for the reshape frame

Zero LLM calls. Read-only intent; the only writes are archive flips on
the IsolatedBrain temp copy to simulate last week's graph.

    ./dev python3 eval/community_reshape_probe.py
"""
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.s2.community_contract import COMMUNITY_DETECTION  # noqa: E402

COVER_PART = 0.25    # a fresh cluster holding >=25% of a stored community is a "part"
COVER_CONTAIN = 0.5  # a stored community >=50% inside one fresh cluster (merge test)
BIRTH_NOVEL = 0.7    # a fresh cluster >=70% outside all stored membership is a birth
GIANT = 40
MIN_STORED = 3       # stored communities with fewer connected members are too sparse to diff


def fresh_partition(brain, config):
    """The decoder's own steps 1-4b, unanchored (no community_state) —
    the whole-graph fresh clustering."""
    from servers.scales.s2 import community_decoder as cd
    dec = cd.CommunityDecoder(brain, config=config)
    rel_to_fam = brain.aspects.primary_edge_map()
    skip = set(cd.ADJACENCY_SKIP_ASPECTS)
    edges_by_node, typed_neighbors = dec._build_typed_adjacency(rel_to_fam, skip)
    pair_z, _degrees, _ = dec._compute_pair_scores(typed_neighbors, edges_by_node)
    direct = dec._get_direct_pairs(edges_by_node)
    clusters = dec._seed_clusters(pair_z, direct)
    valid, corridors, dissolved = dec._validate_clusters(
        clusters, edges_by_node, typed_neighbors)
    valid, absorbed = dec._absorb_subsets(valid)
    part = {cid: set(m) for cid, m in valid.items()}
    return part, set(edges_by_node.keys()), {
        'corridors': len(corridors), 'dissolved': dissolved,
        'absorbed': absorbed}


def load_stored(brain, connected):
    """Stored communities restricted to edge-connected members."""
    from servers.scales.s2 import community_decoder as cd
    dec = cd.CommunityDecoder(brain, config=dict(COMMUNITY_DETECTION))
    out = {}
    for comm in dec._read_community_state():
        mem = set(comm['members']) & connected
        out[comm['id']] = (comm.get('title', '?'), mem, len(comm['members']))
    return out


def diff_against_stored(stored, fresh):
    """Classify every stored community (>= MIN_STORED connected members)
    against the fresh partition."""
    by_member = defaultdict(set)
    for fid, mem in fresh.items():
        for m in mem:
            by_member[m].add(fid)

    res = {'matched': [], 'split': [], 'dispersed': [], 'skipped_sparse': 0}
    for sid, (title, mem, _full) in stored.items():
        if len(mem) < MIN_STORED:
            res['skipped_sparse'] += 1
            continue
        counts = Counter()
        for m in mem:
            for fid in by_member.get(m, ()):
                counts[fid] += 1
        parts = [(fid, c / len(mem)) for fid, c in counts.items()
                 if c / len(mem) >= COVER_PART]
        parts.sort(key=lambda x: -x[1])
        if not parts:
            res['dispersed'].append((sid, title, len(mem)))
            continue
        if len(parts) >= 2:
            res['split'].append((sid, title, len(mem), parts))
            continue
        fid = parts[0][0]
        inter = counts[fid]
        jac = inter / len(mem | fresh[fid])
        res['matched'].append((sid, title, len(mem), jac, parts[0][1]))
    return res


def merge_and_birth(stored, fresh):
    """Merge events (one fresh cluster containing several stored) and
    births (fresh clusters mostly outside stored membership)."""
    all_stored_members = set()
    for _sid, (_t, mem, _f) in stored.items():
        all_stored_members |= mem
    contained = defaultdict(list)
    for sid, (title, mem, _f) in stored.items():
        if len(mem) < MIN_STORED:
            continue
        for fid, fmem in fresh.items():
            if len(mem & fmem) / len(mem) >= COVER_CONTAIN:
                contained[fid].append((sid, title, len(mem)))
    merges = {fid: v for fid, v in contained.items() if len(v) >= 2}
    births = [(fid, len(fmem)) for fid, fmem in fresh.items()
              if len(fmem) >= 4
              and len(fmem - all_stored_members) / len(fmem) >= BIRTH_NOVEL]
    return merges, births


def size_stats(part):
    sizes = sorted((len(m) for m in part.values()), reverse=True)
    if not sizes:
        return 'empty'
    med = sizes[len(sizes) // 2]
    return '%d clusters; median %d, >10: %d, >40: %d, max %d' % (
        len(sizes), med, sum(s > 10 for s in sizes),
        sum(s > 40 for s in sizes), sizes[0])


def main():
    from tests.isolated_brain import IsolatedBrain

    with IsolatedBrain() as env:
        brain = env.brain
        cfg = dict(COMMUNITY_DETECTION)

        fresh_now, connected_now, s_now = fresh_partition(brain, cfg)
        stored = load_stored(brain, connected_now)

        print('=' * 70)
        print('RESHAPE-DIFF FEASIBILITY — fresh whole-graph derivation')
        print('=' * 70)
        print('\nA. populations')
        print('  edge-connected nodes: %d' % len(connected_now))
        print('  fresh partition: %s' % size_stats(fresh_now))
        print('  (corridors %(corridors)d, dissolved %(dissolved)d, '
              'absorbed %(absorbed)d)' % s_now)
        placed = {m for _s, (_t, mem, _f) in stored.items() for m in mem}
        print('  stored: %d live communities; %d of their members '
              'edge-connected' % (len(stored), len(placed)))

        d = diff_against_stored(stored, fresh_now)
        merges, births = merge_and_birth(stored, fresh_now)
        n_scored = len(d['matched']) + len(d['split']) + len(d['dispersed'])
        print('\nB. stored vs fresh (%d scored, %d too sparse)' % (
            n_scored, d['skipped_sparse']))
        for thr in (0.3, 0.5, 0.7):
            k = sum(1 for *_x, jac, _c in d['matched'] if jac >= thr)
            print('  identity survives (matched, Jaccard >= %.1f): '
                  '%d (%.0f%%)' % (thr, k, 100 * k / max(1, n_scored)))
        print('  split into >=2 parts: %d (%.0f%%) | dispersed: %d (%.0f%%) '
              '| merge events: %d | births: %d' % (
                  len(d['split']), 100 * len(d['split']) / max(1, n_scored),
                  len(d['dispersed']),
                  100 * len(d['dispersed']) / max(1, n_scored),
                  len(merges), len(births)))

        print('\nC. the giants under free derivation')
        giants = sorted(((sid, t, mem) for sid, (t, mem, _f) in stored.items()
                         if len(mem) >= GIANT), key=lambda x: -len(x[2]))[:10]
        by_member = defaultdict(set)
        for fid, mem in fresh_now.items():
            for m in mem:
                by_member[m].add(fid)
        for sid, title, mem in giants:
            counts = Counter()
            for m in mem:
                for fid in by_member.get(m, ()):
                    counts[fid] += 1
            tops = sorted((c / len(mem) for c in counts.values()),
                          reverse=True)[:4]
            n_parts = sum(1 for c in counts.values() if c / len(mem) >= 0.10)
            print('  %s (%d conn) "%s": %d parts >=10%%; top coverage %s' % (
                sid, len(mem), title[:44], n_parts,
                ', '.join('%.2f' % t for t in tops) or '-'))

        # D. weekly velocity — archive the last 7 days on the throwaway copy
        cutoff = (datetime.now(timezone.utc)  # clock-ok — eval bookkeeping
                  - timedelta(days=7)).isoformat()
        brain.conn.execute(
            "UPDATE nodes SET archived = 1 WHERE archived = 0 "
            "AND type != 'community' AND created_at > ?", (cutoff,))
        brain.conn.execute(
            "UPDATE edge_relations SET archived = 1 WHERE archived = 0 "
            "AND created_at > ?", (cutoff,))
        brain.conn.commit()
        fresh_old, connected_old, _ = fresh_partition(brain, cfg)

        pseudo = {fid: ('cluster', mem, len(mem))
                  for fid, mem in fresh_old.items()}
        dv = diff_against_stored(pseudo, fresh_now)
        mv, bv = merge_and_birth(pseudo, fresh_now)
        nv = len(dv['matched']) + len(dv['split']) + len(dv['dispersed'])
        stable = sum(1 for *_x, jac, _c in dv['matched'] if jac >= 0.5)
        print('\nD. weekly velocity (fresh@-7d -> fresh@now; %d old clusters '
              'scored)' % nv)
        print('  stable (Jaccard >= 0.5): %d (%.0f%%)' % (
            stable, 100 * stable / max(1, nv)))
        print('  events/week: %d splits, %d merges, %d dispersals, '
              '%d births' % (len(dv['split']), len(mv),
                             len(dv['dispersed']), len(bv)))
        print('  new connected nodes this week: %d' % (
            len(connected_now - connected_old)))


if __name__ == '__main__':
    main()
