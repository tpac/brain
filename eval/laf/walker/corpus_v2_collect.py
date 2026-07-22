"""Corpus-v2 verdict collector — assemble corpus_v2_verdicts.jsonl.

Scans OUT_DIR/corpus_v2_batches/verdicts_batch_*.json (written by the judge
agents, one file per batch — the per-batch checkpoint), joins each batch's
verdicts to its expected keys ORDER-BASED (judges occasionally typo a key
segment — calibration caught 2/40; presentation order is the reliable join),
and writes the merged, resumable-by-key JSONL.

Also reports coverage: which batches are missing / short.

Run:  ./dev python3 eval/laf/walker/corpus_v2_collect.py
"""
import json
import sys

from walker_db import OUT_DIR

BUNDLES = OUT_DIR / 'corpus_v2_bundles.jsonl'
BATCH_DIR = OUT_DIR / 'corpus_v2_batches'
OUT = OUT_DIR / 'corpus_v2_verdicts.jsonl'
BATCH = 25


def main():
    keys = [json.loads(x)['key'] for x in BUNDLES.open()]
    expected = {i // BATCH: keys[i:i + BATCH]
                for i in range(0, len(keys), BATCH)}
    merged, missing, short, repaired = {}, [], [], 0
    for bi in sorted(expected):
        p = BATCH_DIR / ('verdicts_batch_%03d.json' % bi)
        if not p.exists():
            missing.append(bi)
            continue
        try:
            vs = json.loads(p.read_text())
        except Exception as e:
            print('batch %03d unreadable: %s' % (bi, e))
            missing.append(bi)
            continue
        if isinstance(vs, dict):
            vs = vs.get('verdicts', [])
        want = expected[bi]
        if len(vs) != len(want):
            short.append((bi, len(vs), len(want)))
        for i, v in enumerate(vs[:len(want)]):
            if v.get('key') != want[i]:
                v['key_as_returned'] = v.get('key')
                v['key'] = want[i]
                repaired += 1
            v['batch'] = bi
            merged[v['key']] = v
    with OUT.open('w') as f:
        for k in keys:
            if k in merged:
                f.write(json.dumps(merged[k]) + '\n')
    print('merged %d/%d verdicts → %s (keys repaired: %d)'
          % (len(merged), len(keys), OUT, repaired))
    if missing:
        print('MISSING batches: %s' % missing)
    for bi, got, want in short:
        print('SHORT batch %03d: %d/%d' % (bi, got, want))
    return 0


if __name__ == '__main__':
    sys.exit(main())
