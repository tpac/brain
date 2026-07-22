"""Corpus-v2 judge-pass bundle extractor (protocol node 25cea181).

Per gold turn (field_cache_index.json, 2152), emit the judge bundle:
  - the msg (op_text, FULL — not 500-cap)
  - 2-3 prior turns + the turn's own anchor response + 1 turn after
    (the 361692d0 rule — a single cue is 'somewhat worthless')
  - the gold node FULL (title/content/situation/type/age at turn time),
    with the content-graft warning when the node was revised after the
    turn (a0ac8ce4 time-leak guard — candidates.node_revised_after_turn)
  - the turn's Haiku picks (sel=1) with titles, the gold's soft score
  - mechanical telemetry (mix/F0/M_h tie-fair ranks, strong tier, v0
    stratum) — kept in the JSONL for synthesis; the judge renderer
    withholds v0_stratum so semantic verdicts stay independent.

Machinery imported, never re-implemented: Turn (per-msg kernel),
rank_in (tie-fair), zn/lambda_star. Read-only on walker.db + brain.db.

Run:  ./dev python3 eval/laf/walker/corpus_v2_extract.py
Out:  OUT_DIR/corpus_v2_bundles.jsonl
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, WALKER_DB, open_ro, open_brain_ro

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from mesh_fit_probe import Turn                                      # noqa: E402
from lambda_probe import zn, lambda_star                             # noqa: E402
from miss_anatomy import rank_in                                     # noqa: E402

LAM = 0.65
N_BEFORE = 3
CTX_SNIP = 1500       # per context turn text
RESP_SNIP = 2500      # the turn's own anchor response (outcome — rubric 1)
OUT = OUT_DIR / 'corpus_v2_bundles.jsonl'


def iso_days(a, b):
    try:
        return round((datetime.fromisoformat(a.replace('Z', '+00:00'))
                      - datetime.fromisoformat(b.replace('Z', '+00:00'))
                      ).total_seconds() / 86400, 1)
    except Exception:
        return None


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    fields = np.load(OUT_DIR / 'field_cache.npy', mmap_mode='r')
    S = {s: i for i, s in enumerate(idx['slots'])}
    master = idx['master']

    walker = open_ro(WALKER_DB)
    texts = {}
    for sess, epoch, seq, ts, op_t, an_t in walker.execute(
            'SELECT session_id, epoch, seq, ts, op_text, anchor_text '
            'FROM turns'):
        texts[(sess, epoch, seq)] = (ts, op_t or '', an_t or '')
    # gold label provenance guards, keyed (sess, epoch, seq, cand_short)
    cand_meta = {}
    for sess, epoch, seq, cs, created, revised in walker.execute(
            'SELECT session_id, epoch, seq, cand_short, node_created_at, '
            'node_revised_after_turn FROM candidates'):
        cand_meta[(sess, epoch, seq, cs)] = (created, revised)
    walker.close()

    # every node this pass touches: golds + picks
    want = set()
    for t in idx['turns']:
        for i, r in enumerate(t['cand_rows']):
            if r >= 0 and (i == t['gold_i'] or (t.get('sel') and t['sel'][i])):
                want.add(master[r])
    brain = open_brain_ro()
    nodes = {}
    ids = sorted(want)
    for chunk in (ids[i:i + 500] for i in range(0, len(ids), 500)):
        for nid, title, typ, content, created in brain.execute(
                'SELECT id, title, type, content, created_at FROM nodes '
                'WHERE id IN (%s)' % ','.join('?' * len(chunk)), chunk):
            nodes[nid] = {'id': nid, 'title': title, 'type': typ,
                          'content': content, 'created_at': created}
        for nid, val in brain.execute(
                "SELECT node_id, value FROM node_metadata_kv WHERE "
                "key='situation' AND node_id IN (%s)"
                % ','.join('?' * len(chunk)), chunk):
            if nid in nodes:
                nodes[nid]['situation'] = val
    brain.close()

    n_out = n_dead = 0
    with OUT.open('w') as fh:
        for t in idx['turns']:
            sess, epoch, seq = t['key']
            key = '%s/%d/%d' % (sess, epoch, seq)
            gr_row = t['cand_rows'][t['gold_i']]
            gold_short = master[gr_row] if gr_row >= 0 else None
            gold = nodes.get(gold_short) if gold_short else None

            ts, op_t, an_t = texts.get(tuple(t['key']), (t.get('ts'), '', ''))
            before = []
            for pseq in range(seq - 1, -1, -1):
                row = texts.get((sess, epoch, pseq))
                if row and (row[1] or row[2]):
                    before.append({'op': row[1][:CTX_SNIP],
                                   'anchor': row[2][:CTX_SNIP]})
                if len(before) == N_BEFORE:
                    break
            before.reverse()
            after = None
            for nseq in range(seq + 1, seq + 8):
                row = texts.get((sess, epoch, nseq))
                if row and (row[1] or row[2]):
                    after = {'op': row[1][:CTX_SNIP],
                             'anchor': row[2][:CTX_SNIP]}
                    break

            picks = []
            for i, r in enumerate(t['cand_rows']):
                if r >= 0 and t.get('sel') and t['sel'][i] and i != t['gold_i']:
                    nd = nodes.get(master[r])
                    if nd:
                        sv = t['soft'][i]
                        picks.append({'title': nd['title'], 'type': nd['type'],
                                      'soft': round(sv, 3) if sv is not None
                                      else None})

            telemetry, v0 = None, None
            tt = Turn(t, fields, S)
            if tt.gr >= 0 and tt.mh is not None \
                    and not np.isnan(tt.fields[0]).all():
                r_f0 = rank_in(tt.fields[0], tt.gr)
                r_mh = rank_in(tt.mh, tt.gr)
                rk = lambda_star(zn(tt.fields[0]), zn(tt.mh), tt.gr,
                                 grid=np.array([LAM]))
                mix = min(rk.values()) if rk else None
                telemetry = {'mix_rank': mix, 'f0_rank': r_f0,
                             'mh_rank': r_mh, 'strong': tt.strong}
                v0 = ('CUE-SUFF' if r_f0 is not None and r_f0 <= 25 else
                      'MOMENT-DEP' if r_mh is not None and r_mh <= 25 else
                      'NEITHER')

            gold_out = None
            if gold:
                created, revised = cand_meta.get(
                    (sess, epoch, seq, gold_short), (None, None))
                gold_out = dict(gold)
                gold_out['age_days'] = iso_days(ts or t.get('ts') or '',
                                                gold['created_at'])
                gold_out['revised_after_turn'] = bool(revised)
                sv = t['soft'][t['gold_i']]
                gold_out['soft'] = round(sv, 3) if sv is not None else None
            else:
                n_dead += 1

            fh.write(json.dumps({
                'key': key, 'ts': ts or t.get('ts'),
                'op_text': op_t, 'before': before,
                'anchor_response': an_t[:RESP_SNIP], 'after': after,
                'gold': gold_out, 'picks': picks,
                'telemetry': telemetry, 'v0_stratum': v0,
            }) + '\n')
            n_out += 1

    print('wrote %d bundles (%d dead-gold) → %s' % (n_out, n_dead, OUT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
