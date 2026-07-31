"""Inhibition opportunity census — the unmeasured half of edge value.

Typed edges as operation-selectors (id:091c0fb9): correction-aspect edges
carry the inhibition signal cosine structurally cannot (anisotropy — no
negative end). Never measured: how often would suppressing a stale top-5
node / promoting its corrector actually matter on the gold corpus?

Counts, over ALL corpus-v2 valid turns (quality era), under the shipped mix:
  A. RESCUE-SHAPED: gold is correction-linked to a top-5 node — split by
     which end the gold is (gold=corrector: inhibiting the shown stale node
     + promoting the gold is a direct rescue; gold=corrected: inhibition
     would HURT — the count that bounds the op's risk), with the gold's
     rank distance (how far the promotion must carry).
  B. HYGIENE: both ends of a correction edge inside the top-5 together —
     the dedup/inhibit opportunity that doesn't involve the gold.
Correction-aspect verbs come from aspects_v1.json (correction_improvement
member list — read, never hardcoded).

Read-only. Run:  ./dev python3 eval/laf/walker/role_inhibition_census.py
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.insert(0, str(OUT_DIR))
from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn                                         # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CUTOFF = '2026-05-11'
LAM = 0.65
REPORT = OUT_DIR / 'role_inhibition_census.md'


def iso(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '+00:00'))
    except Exception:
        return None


def main():
    aspects = json.loads(
        (REPO / 'servers/scales/s2/aspects_v1.json').read_text())
    entry = aspects.get('correction_improvement') or {}
    corr_verbs = set(entry.get('edge_relations') or [])
    if not corr_verbs:
        raise SystemExit('correction_improvement aspect not found in aspects_v1.json')
    print('correction verbs (%d): %s' % (len(corr_verbs), sorted(corr_verbs)))

    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    verds = {json.loads(x)['key']: json.loads(x)
             for x in (OUT_DIR / 'corpus_v2_verdicts.jsonl').open()}
    bundles = {json.loads(x)['key']: json.loads(x)
               for x in (OUT_DIR / 'corpus_v2_bundles.jsonl').open()}

    b = open_brain_ro()
    # correction edges only, directed: source corrects/supersedes target
    corr_out = defaultdict(list)   # corrector_i -> [(corrected_i, rel, created)]
    corr_in = defaultdict(list)    # corrected_i -> [(corrector_i, rel, created)]
    n_edges = 0
    for src, tgt, rel, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0) AND r.relation IN (%s)"
            % ','.join('?' * len(corr_verbs)), sorted(corr_verbs)):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        corr_out[si].append((ti, rel, created))
        corr_in[ti].append((si, rel, created))
        n_edges += 1
    b.close()
    print('correction edges over master: %d' % n_edges)

    n_turns = 0
    gold_is_corrector = []     # (gold_rank, verb) — rescue-shaped
    gold_is_corrected = []     # (gold_rank, verb) — inhibition would hurt
    hygiene_turns = 0
    hygiene_pairs = Counter()
    by_door = Counter()
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
        n_turns += 1
        turn_dt = iso(bd['ts'])
        top5 = set(int(x) for x in order[:5])
        door = 'door-1' if v['stratum'] == 'cue' else 'door-2'

        def visible(created):
            edt = iso(created)
            return not (turn_dt and edt and edt > turn_dt)

        if gold_rank > 5:
            # gold = corrector of a shown node? (suppress shown, promote gold)
            for (ci, rel, created) in corr_out.get(tt.gr, ()):
                if ci in top5 and visible(created):
                    gold_is_corrector.append((gold_rank, rel))
                    by_door[(door, 'rescue_shaped')] += 1
                    break
            # gold = corrected BY a shown node? (inhibition would bury gold)
            for (ci, rel, created) in corr_in.get(tt.gr, ()):
                if ci in top5 and visible(created):
                    gold_is_corrected.append((gold_rank, rel))
                    by_door[(door, 'would_hurt')] += 1
                    break
        # hygiene: both ends shown together
        pair_found = False
        for si in top5:
            for (ti_, rel, created) in corr_out.get(si, ()):
                if ti_ in top5 and visible(created):
                    hygiene_pairs[rel] += 1
                    pair_found = True
        if pair_found:
            hygiene_turns += 1
            by_door[(door, 'hygiene')] += 1

    def dist(rows):
        ranks = sorted(r for r, _ in rows)
        if not ranks:
            return 'n/a'
        return 'median rank %d · ≤10: %d · ≤25: %d' % (
            ranks[len(ranks) // 2],
            sum(1 for r in ranks if r <= 10),
            sum(1 for r in ranks if r <= 25))

    L = ['# Inhibition opportunity census — correction edges × shipped top-5', '',
         'valid turns: %d · correction edges in graph: %d' % (n_turns, n_edges), '',
         '## A. Rescue-shaped (gold rank>5, correction-linked to a top-5 node)', '',
         '- gold IS the corrector (suppress shown stale → promote gold): '
         '**%d turns** — %s' % (len(gold_is_corrector), dist(gold_is_corrector)),
         '- gold IS the corrected (inhibition would HURT): **%d turns** — %s'
         % (len(gold_is_corrected), dist(gold_is_corrected)),
         '- verbs (rescue-shaped): %s' % dict(Counter(
             r for _, r in gold_is_corrector)),
         '',
         '## B. Hygiene (both ends of a correction edge in top-5 together)', '',
         '- turns: **%d** (%.1f%%) · pair verbs: %s'
         % (hygiene_turns, 100.0 * hygiene_turns / max(n_turns, 1),
            dict(hygiene_pairs)),
         '',
         '## By door', '',
         '| door | rescue-shaped | would-hurt | hygiene |', '|---|---|---|---|']
    for door in ('door-1', 'door-2'):
        L.append('| %s | %d | %d | %d |' % (
            door, by_door[(door, 'rescue_shaped')],
            by_door[(door, 'would_hurt')], by_door[(door, 'hygiene')]))
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
