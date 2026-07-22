"""Corpus-v2 judge batches — render bundles → per-batch markdown files.

Reads corpus_v2_bundles.jsonl (corpus_v2_extract.py), writes
OUT_DIR/corpus_v2_batches/batch_NNN.md, ~BATCH turns each, corpus order
(session-grouped — a judge reads one session's turns coherently).

--keys k1,k2,...  render exactly these keys into calibration_batch_N.md
                  (the 40-turn calibration gate)

The rendering WITHHOLDS v0_stratum and the F0/M_h rank split — the
semantic stratum call must not anchor on the mechanical one. Only the
static-mix rank is shown (rubric 3/4 need hit-vs-miss).
"""
import json
import sys
from pathlib import Path

from walker_db import OUT_DIR

BUNDLES = OUT_DIR / 'corpus_v2_bundles.jsonl'
BATCH_DIR = OUT_DIR / 'corpus_v2_batches'
BATCH = 25
CTX = 800            # chars per side per context turn at render time


def one_line(s, cap):
    return (s or '').replace('\n', ' ⏎ ')[:cap]


def render(b, i):
    L = ['## TURN %d · key=%s · %s' % (i, b['key'], (b['ts'] or '')[:16]), '']
    if b['before']:
        L.append('**Conversation before:**')
        for j, t in enumerate(b['before']):
            off = len(b['before']) - j
            if t['op']:
                L.append('- [t−%d] Tom: %s' % (off, one_line(t['op'], CTX)))
            if t['anchor']:
                L.append('- [t−%d] Anchor: %s' % (off,
                                                  one_line(t['anchor'], CTX)))
        L.append('')
    L.append('**THE MESSAGE (recall fires on this):**')
    L.append('')
    L.append('> ' + one_line(b['op_text'], 4000))
    L.append('')
    L.append('**What Anchor did next (the outcome — for the echo test):** %s'
             % one_line(b['anchor_response'], 1600))
    if b['after']:
        a = b['after']
        if a['op']:
            L.append('- [t+1] Tom: %s' % one_line(a['op'], CTX))
        if a['anchor']:
            L.append('- [t+1] Anchor: %s' % one_line(a['anchor'], CTX))
    L.append('')
    g = b['gold']
    if not g:
        L.append('**GOLD: (node no longer in brain — judge as ambiguous, '
                 'gap="dead node")**')
        L.append('')
        return '\n'.join(L)
    L.append('**THE GOLD (the label under judgment):**')
    L.append('- [%s] %s' % (g['type'], g['title']))
    L.append('- created %s · age at turn %s d · soft %s'
             % ((g['created_at'] or '')[:10], g['age_days'], g['soft']))
    L.append('- content: %s' % one_line(g['content'], 3000))
    if g.get('situation'):
        L.append('- situation: %s' % one_line(g['situation'], 500))
    L.append('')
    if b['picks']:
        L.append('**Haiku picked instead that turn:** '
                 + ' · '.join('[%s] %s' % (p['type'],
                                           one_line(p['title'], 100))
                              for p in b['picks']))
    tel = b.get('telemetry')
    if tel and tel.get('mix_rank') is not None:
        L.append('**Retrieval outcome:** static-mix rank %d → %s'
                 % (tel['mix_rank'],
                    'HIT (top-5)' if tel['mix_rank'] <= 5 else 'MISS'))
    L.append('')
    return '\n'.join(L)


def main():
    keys = None
    if len(sys.argv) > 2 and sys.argv[1] == '--keys':
        keys = [k.strip() for k in sys.argv[2].split(',') if k.strip()]
    bundles = [json.loads(x) for x in BUNDLES.open()]
    BATCH_DIR.mkdir(exist_ok=True)

    if keys:
        by_key = {b['key']: b for b in bundles}
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit('unknown keys: %r' % missing)
        chosen = [by_key[k] for k in keys]
        n_files = 0
        for ci in range(0, len(chosen), BATCH):
            chunk = chosen[ci:ci + BATCH]
            p = BATCH_DIR / ('calibration_batch_%d.md' % (ci // BATCH))
            p.write_text('\n'.join(render(b, ci + i)
                                   for i, b in enumerate(chunk)))
            n_files += 1
        print('calibration: %d turns → %d files in %s'
              % (len(chosen), n_files, BATCH_DIR))
        return 0

    for bi in range(0, len(bundles), BATCH):
        chunk = bundles[bi:bi + BATCH]
        p = BATCH_DIR / ('batch_%03d.md' % (bi // BATCH))
        p.write_text('\n'.join(render(b, bi + i)
                               for i, b in enumerate(chunk)))
    print('%d bundles → %d batch files in %s'
          % (len(bundles), (len(bundles) + BATCH - 1) // BATCH, BATCH_DIR))
    return 0


if __name__ == '__main__':
    sys.exit(main())
