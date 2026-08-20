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


def window_messages(captured):
    """[(turn_n, voice, text, encoded)] for every <other>/<me> in the window."""
    out = []
    # newer captures carry extra attrs between n and encoded (e.g. age="9m ago")
    for m in re.finditer(
            r'<turn n="(\d+)"[^>]*encoded="(true|false)"[^>]*>(.*?)</turn>',
            captured, re.DOTALL):
        n, enc, body = int(m.group(1)), m.group(2) == 'true', m.group(3)
        for voice in ('other', 'me'):
            for mm in re.finditer(
                    r'<%s trace="[0-9a-f]+">(.*?)</%s>' % (voice, voice),
                    body, re.DOTALL):
                text = mm.group(1).strip()
                if text:
                    out.append((n, voice, text[:QUERY_CAP], enc))
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


def aggregate(per_msg_hits, seed_msgs, catalog):
    """turnmax over the chosen seeds, catalog excluded. Offline arithmetic —
    per-message recalls run once, every K composes from the cache."""
    agg = {}
    for key in seed_msgs:
        n, voice = key
        for rank, h in enumerate(per_msg_hits[key]):
            nid = str(h.get('id', ''))[:8]
            if not nid or nid in catalog:
                continue
            s = _score_of(h, rank)
            a = agg.setdefault(nid, {
                'score': 0.0, 'sum': 0.0, 'hits': set(),
                'title': h.get('title', ''), 'type': h.get('type', ''),
                'created': str(h.get('created_at', ''))[:10]})
            a['score'] = max(a['score'], s)              # turnmax lens
            a['sum'] += s                                # turnsum lens (door-2's
            a['hits'].add('t%d/%s' % (n, voice))         # in-field sum, approximated)
    return sorted(agg.items(), key=lambda kv: -kv[1]['score'])


def _shift(day, delta_days):
    import datetime
    d = datetime.date.fromisoformat(day) + datetime.timedelta(days=delta_days)
    return d.isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('--limit-per-query', type=int, default=40,
                    help='per-message fetch BEFORE exclusion — must exceed '
                         'the catalog-hot head or novel candidates starve '
                         '(production fix: exclude inside recall, pre-limit)')
    ap.add_argument('--ks', default='1,2,4,unenc,all',
                    help='comma list: N = last-N turns of the window, '
                         '"unenc" = unencoded turns only, "all" = whole window')
    ap.add_argument('--needle', default='78983ba6')
    ap.add_argument('--watch', default='78983ba6,d827d22f,accd8be6,b792b20e',
                    help='ids whose retrievability we track under every lens')
    args = ap.parse_args()

    chain, _short, _stop = parse_chain(args.capture)
    with open(args.capture) as f:
        captured = f.read()
    catalog = catalog_ids_of(captured)
    msgs = window_messages(captured)
    turns = sorted({n for n, _v, _t, _e in msgs})
    unenc_turns = sorted({n for n, _v, _t, e in msgs if not e})
    print('%s: window turns %s (unencoded: %s), %d messages, %d catalog ids'
          % (chain, turns, unenc_turns, len(msgs), len(catalog)))

    os.environ['BRAIN_RECALL_VARIANT'] = 'laf_v1'  # as_of requires LAF
    from isolated_brain import IsolatedBrain
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        _sid, run_ts = resolve_run(brain, chain)
        print('as_of = %s | fetch %d/message, exclusion post-fetch\n'
              % (run_ts, args.limit_per_query))

        per_msg_hits, texts = {}, {}
        for n, voice, text, _e in msgs:
            key = (n, voice)
            r = brain.recall(query=text, limit=args.limit_per_query,
                             source='encode_moment_probe', as_of=run_ts)
            # a turn can carry several messages per voice — keep the union
            per_msg_hits.setdefault(key, []).extend(r.get('results') or [])
            texts[key] = text

        week_ago = _shift(run_ts[:10], -7)
        for spec in [s.strip() for s in args.ks.split(',') if s.strip()]:
            if spec == 'all':
                chosen = set(turns)
            elif spec == 'unenc':
                chosen = set(unenc_turns)
            else:
                chosen = set(turns[-int(spec):])
            seeds = [k for k in per_msg_hits if k[0] in chosen]
            ranked = aggregate(per_msg_hits, seeds, catalog)
            needle_rank = next((i for i, (nid, _a) in enumerate(ranked, 1)
                                if nid == args.needle), None)
            n_fresh = {band: sum(1 for _nid, a in ranked[:band]
                                 if a['created'] >= week_ago)
                       for band in (5, 10, 20)}
            print('═══ K=%s → turns %s (%d seed messages) | candidates=%d | '
                  'needle %s rank=%s | fresh(≤7d) in top5/10/20 = %d/%d/%d'
                  % (spec, sorted(chosen), len(seeds), len(ranked),
                     args.needle, needle_rank, n_fresh[5], n_fresh[10],
                     n_fresh[20]))
            for i, (nid, a) in enumerate(ranked[:10], 1):
                mark = ' ◄ needle' if nid == args.needle else ''
                print('  %2d. %.4f %s [%s] %s (%s)%s'
                      % (i, a['score'], nid, a['type'][:9], a['title'][:64],
                         a['created'], mark))
            if needle_rank and needle_rank > 10:
                a = dict(ranked)[args.needle]
                print('  ...%2d. %.4f %s %s ◄ needle'
                      % (needle_rank, a['score'], args.needle, a['title'][:60]))
            # turnsum lens — the wired stack's composition family, outside-field proxy
            by_sum = sorted(ranked, key=lambda kv: -kv[1]['sum'])
            sum_rank = next((i for i, (nid, _a) in enumerate(by_sum, 1)
                             if nid == args.needle), None)
            print('  turnsum lens: needle rank=%s | top5: %s'
                  % (sum_rank, ', '.join('%s(%.2f)' % (nid, a['sum'])
                                         for nid, a in by_sum[:5])))
            watch = [w.strip() for w in args.watch.split(',') if w.strip()]
            pos_max = {nid: i for i, (nid, _a) in enumerate(ranked, 1)}
            pos_sum = {nid: i for i, (nid, _a) in enumerate(by_sum, 1)}
            for w in watch:
                print('  watch %s: turnmax=%s turnsum=%s'
                      % (w, pos_max.get(w, 'NOT RETRIEVED'),
                         pos_sum.get(w, 'NOT RETRIEVED')))
            print()

        # ── blob lens: the whole unencoded window as ONE query (Tom's test:
        # does a single moment-cosine find what per-message seeds might not?)
        blob = '\n'.join(t for n, _v, t, e in msgs if not e)[:6000]
        r = brain.recall(query=blob, limit=200,
                         source='encode_moment_probe_blob', as_of=run_ts)
        hits = [(str(h.get('id', ''))[:8], h) for h in (r.get('results') or [])]
        novel = [(nid, h) for nid, h in hits if nid and nid not in catalog]
        print('═══ BLOB lens: all unencoded turns as one query (%d chars) | '
              '%d hits, %d beyond-catalog' % (len(blob), len(hits), len(novel)))
        for i, (nid, h) in enumerate(novel[:5], 1):
            print('   %d. %.4f %s [%s] %s' % (i, _score_of(h, i), nid,
                  str(h.get('type', ''))[:9], str(h.get('title', ''))[:64]))
        pos_blob = {nid: i for i, (nid, _h) in enumerate(novel, 1)}
        pos_raw = {nid: i for i, (nid, _h) in enumerate(hits, 1)}
        for w in [w.strip() for w in args.watch.split(',') if w.strip()]:
            print('   watch %s: blob-novel rank=%s (raw=%s)'
                  % (w, pos_blob.get(w, 'NOT RETRIEVED'),
                     pos_raw.get(w, 'NOT RETRIEVED')))


if __name__ == '__main__':
    main()
