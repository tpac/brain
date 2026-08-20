"""Encode-time moment recall — the eyeball probe (door-1 approximation).

Answers: if encoder assembly ran recall over the window's UNENCODED messages
(both voices, one query per message — never a blob), excluded the catalog,
and aggregated turnmax — what would the <beyond_catalog> stubs have been?

Door-2 shape, door-1 parts: uniform decay x turnmax over per-message seeds
is the degenerate cell of the moment grid; when moments recall ships, the
provider swaps under the same call site.

Runs recall against an IsolatedBrain COPY (recall mutates access/fatigue —
never probe the live daemon) with `as_of` pinned to the run's capture
instant, so the candidate field is the brain as the run saw it — today's
nodes (which discuss these very ids) cannot leak in.

Usage:
    ./dev python3 eval/encode_moment_recall_probe.py <payloads/.../000-prompt.md>
        [--limit-per-query 15] [--top 12]
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoder_prompt_reassembly import catalog_ids_of, parse_chain, resolve_run  # noqa: E402

QUERY_CAP = 1500  # embedding sanity; the seed is the message, not the essay


def unencoded_messages(captured):
    """[(turn_n, voice, text)] for every <other>/<me> in encoded="false" turns."""
    out = []
    for m in re.finditer(
            r'<turn n="(\d+)" encoded="false">(.*?)</turn>', captured, re.DOTALL):
        n, body = int(m.group(1)), m.group(2)
        for voice in ('other', 'me'):
            for mm in re.finditer(
                    r'<%s trace="[0-9a-f]+">(.*?)</%s>' % (voice, voice),
                    body, re.DOTALL):
                text = mm.group(1).strip()
                if text:
                    out.append((n, voice, text[:QUERY_CAP]))
    return out


def _score_of(hit, rank):
    # laf_v1 exposes the settled field as effective_activation (cosine lane
    # visible separately as embedding_similarity). Booleans are ints in
    # Python — earlier version of this probe matched `archived=False` as 0.0
    # and flattened every score; hence the explicit key order + bool guard.
    for k in ('effective_activation', 'embedding_similarity', 'score', '_score'):
        v = hit.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return 1.0 / (rank + 1)  # rank fallback if the mode exposes no score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('--limit-per-query', type=int, default=15)
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()

    chain, _short, _stop = parse_chain(args.capture)
    with open(args.capture) as f:
        captured = f.read()
    catalog = catalog_ids_of(captured)
    msgs = unencoded_messages(captured)
    print('%s: %d unencoded messages (%d turns), %d catalog ids'
          % (chain, len(msgs), len({n for n, _v, _t in msgs}), len(catalog)))

    os.environ['BRAIN_RECALL_VARIANT'] = 'laf_v1'  # as_of requires LAF
    from isolated_brain import IsolatedBrain
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        _sid, run_ts = resolve_run(brain, chain)
        print('as_of = %s (the run\'s capture instant)\n' % run_ts)

        # node_id -> {'score','turns','hit_titles'} — turnmax aggregation
        agg = {}
        for n, voice, text in msgs:
            r = brain.recall(query=text, limit=args.limit_per_query,
                             source='encode_moment_probe', as_of=run_ts)
            hits = r.get('results') or []
            for rank, h in enumerate(hits):
                nid = str(h.get('id', ''))[:8]
                if not nid:
                    continue
                s = _score_of(h, rank)
                a = agg.setdefault(nid, {
                    'score': 0.0, 'hits': [],
                    'title': h.get('title', ''), 'type': h.get('type', ''),
                    'created': str(h.get('created_at', ''))[:10]})
                a['score'] = max(a['score'], s)          # turnmax, never sum
                a['hits'].append('t%d/%s' % (n, voice))

        admitted = {k: v for k, v in agg.items() if k not in catalog}
        excluded = {k: v for k, v in agg.items() if k in catalog}
        print('recall pulled %d distinct nodes: %d already in catalog '
              '(excluded — the novel-only filter at work), %d beyond-catalog\n'
              % (len(agg), len(excluded), len(admitted)))

        print('BEYOND-CATALOG CANDIDATES (turnmax rank — the would-be stubs):')
        ranked = sorted(admitted.items(), key=lambda kv: -kv[1]['score'])
        for i, (nid, a) in enumerate(ranked[:args.top], 1):
            print('%3d. %.4f  %s  [%s] %s  (created %s)\n'
                  '            hit by: %s'
                  % (i, a['score'], nid, a['type'], a['title'][:76],
                     a['created'], ', '.join(sorted(set(a['hits'])))))
        if len(ranked) > args.top:
            print('  … %d more below rank %d' % (len(ranked) - args.top, args.top))

        print('\nTOP EXCLUDED (sanity — should be the hot catalog nodes):')
        for nid, a in sorted(excluded.items(), key=lambda kv: -kv[1]['score'])[:5]:
            print('     %.4f  %s  %s' % (a['score'], nid, a['title'][:70]))


if __name__ == '__main__':
    main()
