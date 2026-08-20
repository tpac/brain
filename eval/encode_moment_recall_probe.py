"""Encode-time moment recall — the eyeball probe (door-1 approximation).

Answers: if encoder assembly ran recall over the window's messages (both
voices, one query per message — never a blob: measured dead, needle absent
from the top-200 of a whole-window query), excluded the catalog, and
aggregated turnmax — what would the <memories_beyond_catalog> stubs be?

Turnmax (a node's score is its best single-message match) is the measured
composition for this moment type: turnsum buried the needle at rank 18-20 by
rewarding broad weak resonance over one hard match. Both verdicts and the
K-sweep live in the brain ([thread:encoder-staleness]); this probe keeps only
the winning shape.

Runs recall against an IsolatedBrain COPY (recall mutates access/fatigue —
never probe the live daemon) with `as_of` pinned to the run's capture
instant, so the candidate field is the brain as the run saw it.

Usage:
    ./dev python3 eval/encode_moment_recall_probe.py <payloads/.../000-prompt.md>
        [--ks 1,2,4,unenc,all] [--watch id1,id2,...] [--limit-per-query 40]

    --ks     lookback lenses: N = last-N window turns, 'unenc' = unencoded
             turns only (the production choice), 'all' = whole window
    --watch  ids whose rank you are tracking; the first is the headline
             needle. No default — pass the ids your investigation cares about.
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

from encoder_prompt_reassembly import (catalog_ids_of, parse_chain,  # noqa: E402
                                       resolve_run, section)

QUERY_CAP = 1500  # embedding sanity; the seed is the message, not the essay


def window_messages(captured):
    """[(turn_n, voice, text, encoded)] for every <other>/<me> in the window.

    Parses the <timeline> section ONLY — catalog node bodies quote the turn
    render format verbatim, so a whole-capture scan reads documentation as
    data. A turn with no encoded= attribute is real (the renderer omits it
    for orphan/link-less turns) and counts as unencoded.
    """
    timeline = section(captured, 'timeline')
    if not timeline:
        raise SystemExit('capture has no <timeline> section — this probe '
                         'requires a lived-render capture (a whole-file scan '
                         'would read catalog-quoted turn markup as data)')
    out = []
    for m in re.finditer(
            r'<turn n="(\d+)"([^>]*)>(.*?)</turn>', timeline, re.DOTALL):
        n, attrs, body = int(m.group(1)), m.group(2), m.group(3)
        enc = 'encoded="true"' in attrs
        for voice in ('other', 'me'):
            for mm in re.finditer(
                    r'<%s trace="[0-9a-f]+">(.*?)</%s>' % (voice, voice),
                    body, re.DOTALL):
                text = mm.group(1).strip()
                if text:
                    out.append((n, voice, text[:QUERY_CAP], enc))
    return out


def _score_of(hit):
    # laf_v1 (which this probe pins) sets effective_activation on every
    # result; booleans are ints in Python, so guard the type.
    v = hit.get('effective_activation')
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def aggregate(per_msg_hits, seed_msgs, catalog):
    """turnmax over the chosen seeds, catalog excluded. Offline arithmetic —
    per-message recalls run once, every K composes from the cache. Returns
    (ranked novel list, excluded-catalog-hit count) — the excluded count is
    the sanity signal that the novel-only filter actually fired."""
    agg, excluded = {}, set()
    for key in seed_msgs:
        for h in per_msg_hits[key]:
            nid = str(h.get('id', ''))[:8]
            if not nid:
                continue
            if nid in catalog:
                excluded.add(nid)
                continue
            a = agg.setdefault(nid, {
                'score': 0.0,
                'title': h.get('title', ''), 'type': h.get('type', ''),
                'created': str(h.get('created_at', ''))[:10]})
            a['score'] = max(a['score'], _score_of(h))   # turnmax, never sum
    return (sorted(agg.items(), key=lambda kv: -kv[1]['score']),
            len(excluded))


def parse_ks(spec_str, turns, unenc_turns):
    """[(label, chosen-turn-set)] — validated BEFORE any recall is paid for."""
    out = []
    for spec in [s.strip() for s in spec_str.split(',') if s.strip()]:
        if spec == 'all':
            out.append((spec, set(turns)))
        elif spec == 'unenc':
            out.append((spec, set(unenc_turns)))
        else:
            try:
                k = int(spec)
            except ValueError:
                raise SystemExit('--ks token %r is not an int, "unenc" or '
                                 '"all"' % spec)
            if k < 1:
                raise SystemExit('--ks values must be >= 1, got %d' % k)
            out.append((spec, set(turns[-k:])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('--limit-per-query', type=int, default=40,
                    help='per-message fetch BEFORE exclusion — must exceed '
                         'the catalog-hot head or novel candidates starve '
                         '(production fix: exclude inside recall, pre-limit)')
    ap.add_argument('--ks', default='unenc',
                    help='comma list: N = last-N turns, "unenc", "all"')
    ap.add_argument('--watch', default='',
                    help='comma list of node ids to rank-track; first is the '
                         'headline needle')
    args = ap.parse_args()
    watch = [w.strip() for w in args.watch.split(',') if w.strip()]

    chain, _short, _stop = parse_chain(args.capture)
    with open(args.capture) as f:
        captured = f.read()
    catalog = catalog_ids_of(captured)
    msgs = window_messages(captured)
    turns = sorted({n for n, _v, _t, _e in msgs})
    unenc_turns = sorted({n for n, _v, _t, e in msgs if not e})
    lenses = parse_ks(args.ks, turns, unenc_turns)
    needed_turns = set().union(*(chosen for _l, chosen in lenses))
    print('%s: window turns %s (unencoded: %s), %d messages, %d catalog ids'
          % (chain, turns, unenc_turns, len(msgs), len(catalog)))

    os.environ['BRAIN_RECALL_VARIANT'] = 'laf_v1'  # as_of requires LAF
    from isolated_brain import IsolatedBrain
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        _sid, run_ts = resolve_run(brain, chain)
        print('as_of = %s | fetch %d/message, exclusion post-fetch\n'
              % (run_ts, args.limit_per_query))

        per_msg_hits = {}
        for n, voice, text, _e in msgs:
            if n not in needed_turns:
                continue  # no requested lens uses this turn — don't pay for it
            r = brain.recall(query=text, limit=args.limit_per_query,
                             source='encode_moment_probe', as_of=run_ts)
            # a turn can carry several messages per voice — keep the union
            per_msg_hits.setdefault((n, voice), []).extend(
                r.get('results') or [])

        for label, chosen in lenses:
            seeds = [k for k in per_msg_hits if k[0] in chosen]
            ranked, n_excl = aggregate(per_msg_hits, seeds, catalog)
            pos = {nid: i for i, (nid, _a) in enumerate(ranked, 1)}
            needle = watch[0] if watch else None
            print('═══ K=%s → turns %s (%d seed messages) | %d beyond-catalog '
                  'candidates, %d catalog hits excluded%s'
                  % (label, sorted(chosen), len(seeds), len(ranked), n_excl,
                     (' | needle %s rank=%s'
                      % (needle, pos.get(needle, 'NOT RETRIEVED'))
                      if needle else '')))
            for i, (nid, a) in enumerate(ranked[:10], 1):
                mark = ' ◄ needle' if nid == needle else ''
                print('  %2d. %.4f %s [%s] %s (%s)%s'
                      % (i, a['score'], nid, a['type'][:9], a['title'][:64],
                         a['created'], mark))
            for w in watch:
                print('  watch %s: rank=%s'
                      % (w, pos.get(w, 'NOT RETRIEVED')))
            print()


if __name__ == '__main__':
    main()
