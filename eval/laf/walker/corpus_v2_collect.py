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
    # v3 OVERRIDES: rejudge (v2-half echoes re-run on v3) and audit (Opus
    # blind pass) supersede the original verdicts_batch_* for their keys.
    # Loaded LAST so they win; keyed by the key field (order already fixed
    # at write time via the calibration/audit key lists). rejudge > audit
    # if both touched a key (rejudge is the production-rubric verdict).
    overrides, ov_repaired = {}, 0
    for tag, glob, keyfile in (
            ('audit', 'verdicts_audit_%03d.json',
             'corpus_v2_audit_keys.txt'),
            ('rejudge', 'verdicts_rejudge_%03d.json',
             'corpus_v2_rejudge_keys.txt')):
        kl = (OUT_DIR / keyfile).read_text().split(',')
        for bi in range((len(kl) + BATCH - 1) // BATCH):
            p = BATCH_DIR / (glob % bi)
            if not p.exists():
                continue
            vs = json.loads(p.read_text())
            want = kl[bi * BATCH:(bi + 1) * BATCH]
            for i, v in enumerate(vs[:len(want)]):
                if v.get('key') != want[i]:
                    ov_repaired += 1
                v['key'] = want[i]          # order-join, authoritative
                v['rubric'] = 'v3'
                v['source'] = tag
                overrides[want[i]] = v      # rejudge loaded last → wins
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
            v.setdefault('rubric', 'v2' if bi <= 51 else 'v3')
            v.setdefault('source', 'batch')
            merged[v['key']] = v
    # apply v3 overrides (carry the original batch index for stratum/age joins)
    applied = 0
    for k, ov in overrides.items():
        if k in merged:
            ov['batch'] = merged[k].get('batch')
            merged[k] = ov
            applied += 1
    with OUT.open('w') as f:
        for k in keys:
            if k in merged:
                f.write(json.dumps(merged[k]) + '\n')
    print('merged %d/%d verdicts → %s (keys repaired: %d, ov-repaired: %d, '
          'v3 overrides applied: %d)'
          % (len(merged), len(keys), OUT, repaired, ov_repaired, applied))
    if missing:
        print('MISSING batches: %s' % missing)
    for bi, got, want in short:
        print('SHORT batch %03d: %d/%d' % (bi, got, want))
    return 0


if __name__ == '__main__':
    sys.exit(main())
