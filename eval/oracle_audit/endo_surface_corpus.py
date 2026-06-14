#!/usr/bin/env python3
"""ENDO-SURFACE CORPUS (increment 1) — the soft-surface instrument substrate.

Forks the oracle-audit moment→candidate pattern for the ENDO surface (Stop tap,
SOFT-SURFACE-DESIGN.md): the cue is Anchor's OWN assistant turn, not the
operator's prompt. Produces a *labelable corpus draft* — each blind-sampled
moment + its actual next turn + the candidates that survive the §7 awareness
filters — for operator labeling. NO LLM judge yet; that's increment 2 (the gold
judge + uptake lens), built once we have operator labels to calibrate against.

§7 awareness suppression (the load-bearing filter — the blind test showed the
dominant noise is "already in my awareness", not irrelevance):
  - TIME-CORRECT : drop candidates created >= the moment. Kills the
                   encoded-from-this-turn echo AND any future node (the §16
                   hindsight guard — replay must not credit itself).
  - ECHO         : drop candidates created earlier in the moment's OWN session
                   (encoder output from my own recent turns).
  - CITED        : drop candidates whose id-prefix or title words already appear
                   in the turn text (a prior I already invoked — not additive).

Blind moment selection: assistant_message S0 turns from sessions strictly OLDER
than today, picked by fixed stride (no cherry-picking — id+time before content).
Daemon-safe: runs against an IsolatedBrain copy, never the live DB.

Usage: ./dev python3 eval/oracle_audit/endo_surface_corpus.py [N]
Writes endo_surface_corpus.json + prints a summary.
"""
import sys
import os
import json
import re

ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RECALL_K = 8
CITED_OVERLAP = 0.6            # title-word overlap fraction → treat as cited
OLDER_THAN = '2026-06-13'      # blind sample from sessions before this (settled, unfamiliar)
OUT = f'{ROOT}/eval/oracle_audit/endo_surface_corpus.json'


def _words(text):
    return set(re.findall(r'[a-z0-9_]{4,}', (text or '').lower()))


def _meta_content(row):
    """Trace metadata may be a dict or a JSON string; pull .content either way."""
    m = row.get('metadata')
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except Exception:
            return row.get('summary', '') or ''
    if isinstance(m, dict):
        return m.get('content') or row.get('summary', '') or ''
    return row.get('summary', '') or ''


def _next_turns(brain, session_id, ts, n=2):
    """The turn(s) right after the moment — the uptake window."""
    try:
        from servers.scales.s0.conversation import get_conversation_around
        win = get_conversation_around(brain, session_id=session_id, timestamp=ts,
                                      before=0, after=n)
        return [{'role': t.get('role'), 'content': (t.get('content') or '')[:400],
                 'ts': t.get('timestamp', '')} for t in win
                if t.get('timestamp', '') > ts][:n]
    except Exception as e:
        return [{'role': 'error', 'content': 'next-turn fetch failed: %s' % e, 'ts': ''}]


with IsolatedBrain() as env:
    b = env.brain
    tdal = b._trace_dal

    # ── 1. Blind moment selection (id+time before content) ──
    rows = tdal.get_by_ref_type('assistant_message', scale='s0',
                                hours=None, limit=4000)
    older = [r for r in rows if (r.get('created_at') or '')[:10] < OLDER_THAN
             and (_meta_content(r) or '').strip()]
    older.sort(key=lambda r: r.get('created_at') or '')
    if not older:
        print('No older assistant_message moments found.')
        sys.exit(1)

    # even stride across the chronological list — blind
    step = max(1, len(older) // N)
    picks = older[::step][:N]
    print('[endo-corpus] %d assistant turns older than %s; sampling %d (stride %d)'
          % (len(older), OLDER_THAN, len(picks), step))

    # session-start cache (for echo detection)
    session_start = {}

    def _sess_start(sid):
        if sid not in session_start:
            ev = tdal.get_recent(scale='s0', session_id=sid, limit=1, hours=None) \
                if hasattr(tdal, 'get_recent') else []
            # fall back: min created_at over this session's traces
            try:
                row = b.logs_conn.execute(
                    'SELECT MIN(created_at) FROM trace_events WHERE session_id=?',
                    (sid,)).fetchone()
                session_start[sid] = (row[0] if row and row[0] else '')
            except Exception:
                session_start[sid] = ''
        return session_start[sid]

    corpus = []
    agg = {'candidates': 0, 'kept': 0, 'time': 0, 'echo': 0, 'cited': 0,
           'silent_moments': 0}

    for i, m in enumerate(picks):
        cue = _meta_content(m)
        ts = m.get('created_at') or ''
        sid = m.get('session_id') or ''
        cue_words = _words(cue)
        sstart = _sess_start(sid)

        try:
            res = b.recall(cue[:2000], limit=RECALL_K)
            cands = res.get('results', []) if isinstance(res, dict) else []
        except Exception as e:
            cands = []
            print('  [moment %d] recall failed: %s' % (i, e))

        kept, drop = [], {'time': 0, 'echo': 0, 'cited': 0}
        for c in cands:
            cid = c.get('id', '')
            ctitle = c.get('title', '')
            cts = c.get('created_at') or ''
            agg['candidates'] += 1
            # TIME-CORRECT: created at/after the moment (incl. encoded-from-this-turn)
            if cts and cts >= ts:
                drop['time'] += 1
                continue
            # ECHO: created earlier in the moment's own session
            if sstart and sstart <= cts < ts:
                drop['echo'] += 1
                continue
            # CITED: id-prefix or title already in the turn text
            tw = _words(ctitle)
            overlap = (len(tw & cue_words) / len(tw)) if tw else 0.0
            if (cid[:8] and cid[:8] in cue) or overlap >= CITED_OVERLAP:
                drop['cited'] += 1
                continue
            kept.append({'id': cid, 'title': ctitle, 'created_at': cts,
                         'score': c.get('score') or c.get('final_score')})

        for k in ('time', 'echo', 'cited'):
            agg[k] += drop[k]
        agg['kept'] += len(kept)
        if not kept:
            agg['silent_moments'] += 1

        corpus.append({
            'idx': i,
            'moment_trace': m.get('id'),
            'date': ts[:16],
            'session': sid[:8],
            'cue': cue[:500],
            'next_turns': _next_turns(b, sid, ts),
            'candidates_kept': kept,           # ← operator labels these: glad / inert
            'dropped': drop,
            'label': None,                     # operator fills: which kept id is "glad", or "silence"
        })

    json.dump({'generated_for': 'endo_surface', 'n': len(corpus),
               'recall_k': RECALL_K, 'filters': ['time', 'echo', 'cited'],
               'aggregate': agg, 'moments': corpus},
              open(OUT, 'w'), indent=2)

    # ── summary ──
    print('\n=== ENDO-SURFACE CORPUS (increment 1) ===')
    print('moments: %d | candidates: %d | kept: %d (%.1f/moment)'
          % (len(corpus), agg['candidates'], agg['kept'],
             agg['kept'] / max(1, len(corpus))))
    print('dropped — time(echo-of-turn+future): %d | echo(same-session): %d | cited: %d'
          % (agg['time'], agg['echo'], agg['cited']))
    print('moments with 0 survivors (silence candidates): %d/%d'
          % (agg['silent_moments'], len(corpus)))
    print('wrote %s' % OUT)
    print('\nNEXT: operator labels candidates_kept per moment (glad / inert / silence);')
    print('increment 2 adds the Sonnet gold-judge + uptake-vs-next-turn lens,')
    print('calibrated against those labels.')
