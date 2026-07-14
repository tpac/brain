"""Gold-exclusion manifest — §20.3, Stage 1 first artifact.

Maps each frozen gold-24 cue to its source turn in the trace substrate
(session_id + stop), so the walker can exclude gold cue turns AND their full
sessions BY CONSTRUCTION (§19 risk 6, §20.3). The gold cards carry `cutoff`
timestamps but no session ids — this script recovers the join once and commits
it as `gold_manifest.json`.

Match method: for each cue, scan s0 conversational trace rows (user_message /
assistant_message per the cue's speaker) inside a ±6h window around the card's
cutoff, and compare normalized text prefixes both directions. Best match =
longest prefix agreement, ties broken by timestamp proximity to cutoff.

Read-only: opens brain_logs.db with mode=ro (the dashboard precedent — never a
writer against the live DBs).

Run:  ./dev python3 eval/laf/walker/gold_manifest.py
Exit: 0 all 24 matched · 2 any unmatched (manifest still written, unmatched
flagged — the walker build refuses a manifest with unmatched cues).
"""
import ast
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GOLD_DIR = REPO / 'eval' / 'oracle_audit' / 'gold_remint'
OUT_PATH = Path(__file__).resolve().parent / 'gold_manifest.json'

WINDOW_HOURS = 6
PREFIX_CHARS = 100
SPEAKER_REF = {'OPERATOR': 'user_message', 'ANCHOR': 'assistant_message'}


def logs_db_path():
    db_dir = os.environ.get('BRAIN_DB_DIR') or str(Path.home() / 'AgentsContext' / 'brain')
    return Path(db_dir) / 'brain_logs.db'


def norm(text):
    return re.sub(r'\s+', ' ', (text or '')).strip().lower()


def parse_cue_field(raw):
    """moments.json stores the cue as a stringified {'speaker','text'} dict."""
    if isinstance(raw, dict):
        return raw.get('speaker', ''), raw.get('text', '')
    try:
        d = ast.literal_eval(raw)
        return d.get('speaker', ''), d.get('text', '')
    except (ValueError, SyntaxError):
        m = re.search(r"'speaker':\s*'(\w+)'", raw or '')
        t = re.search(r"'text':\s*[\"'](.+)", raw or '', re.S)
        return (m.group(1) if m else ''), (t.group(1) if t else raw or '')


def stop_of(chain_id):
    tail = str(chain_id or '').rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else None


def prefix_agreement(a, b, limit=PREFIX_CHARS):
    """Length of agreement between two normalized prefixes (either direction)."""
    a, b = a[:limit], b[:limit]
    if not a or not b:
        return 0
    if a.startswith(b) or b.startswith(a):
        return min(len(a), len(b))
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def match_cue(conn, cue_id, speaker, text, cutoff_iso):
    ref_type = SPEAKER_REF.get(speaker)
    if not ref_type:
        return {'cue_id': cue_id, 'matched': False, 'reason': 'unknown speaker %r' % speaker}
    cutoff = datetime.fromisoformat(cutoff_iso)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    lo = (cutoff - timedelta(hours=WINDOW_HOURS)).isoformat()
    hi = (cutoff + timedelta(hours=WINDOW_HOURS)).isoformat()
    target = norm(text)
    rows = conn.execute(
        "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
        "WHERE scale='s0' AND ref_type=? AND created_at BETWEEN ? AND ?",
        (ref_type, lo, hi)).fetchall()
    best = None
    for session_id, chain_id, created_at, meta_raw in rows:
        try:
            content = norm(json.loads(meta_raw or '{}').get('content', ''))
        except (ValueError, TypeError):
            continue
        score = prefix_agreement(target, content)
        # Floor: long cues need 40 normalized chars of agreement; cues shorter
        # than the floor must match their ENTIRE text (stricter — a 21-char
        # "just commit and merge" matches exactly or not at all).
        if score < min(40, len(target)):
            continue
        row_ts = datetime.fromisoformat(created_at)
        if row_ts.tzinfo is None:
            row_ts = row_ts.replace(tzinfo=timezone.utc)
        delta_s = abs((row_ts - cutoff).total_seconds())
        key = (-score, delta_s)
        if best is None or key < best['key']:
            best = {'key': key, 'session_id': session_id, 'chain_id': chain_id,
                    'created_at': created_at, 'score': score, 'ts_delta_s': int(delta_s)}
    if best is None:
        return {'cue_id': cue_id, 'matched': False, 'speaker': speaker,
                'reason': 'no row above agreement floor in ±%dh window' % WINDOW_HOURS,
                'candidates_in_window': len(rows)}
    return {'cue_id': cue_id, 'matched': True, 'speaker': speaker,
            'session_id': best['session_id'], 'stop': stop_of(best['chain_id']),
            'chain_id': best['chain_id'], 'turn_ts': best['created_at'],
            'prefix_agreement': best['score'], 'ts_delta_s': best['ts_delta_s']}


def main():
    frozen = json.loads((GOLD_DIR / 'frozen_gold_24.json').read_text())
    moments = {m['cue_id']: m for m in json.loads((GOLD_DIR / 'moments.json').read_text())}
    missing_cards = sorted(set(frozen) - set(moments))
    if missing_cards:
        print('FATAL: frozen cues missing from moments.json: %s' % missing_cards)
        return 2

    db = logs_db_path()
    conn = sqlite3.connect('file:%s?mode=ro' % db, uri=True)
    try:
        entries = []
        for cue_id in sorted(frozen):
            card = moments[cue_id]
            speaker, text = parse_cue_field(card.get('cue'))
            entries.append(match_cue(conn, cue_id, speaker, text, card['cutoff']))
    finally:
        conn.close()

    matched = [e for e in entries if e.get('matched')]
    unmatched = [e for e in entries if not e.get('matched')]
    manifest = {
        'built_from': {'gold': 'eval/oracle_audit/gold_remint/frozen_gold_24.json',
                       'moments': 'eval/oracle_audit/gold_remint/moments.json',
                       'window_hours': WINDOW_HOURS, 'prefix_chars': PREFIX_CHARS},
        'cues': entries,
        'excluded_sessions': sorted({e['session_id'] for e in matched}),
        'matched': len(matched), 'unmatched': len(unmatched),
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2) + '\n')

    print('gold manifest: %d/%d cues matched -> %s' % (len(matched), len(entries), OUT_PATH))
    print('excluded sessions: %d' % len(manifest['excluded_sessions']))
    for e in matched:
        print('  %s -> %s stop=%s  (agree=%d, dt=%ds)' % (
            e['cue_id'], e['session_id'][:8], e['stop'], e['prefix_agreement'], e['ts_delta_s']))
    for e in unmatched:
        print('  UNMATCHED %s: %s' % (e['cue_id'], e.get('reason')))
    return 2 if unmatched else 0


if __name__ == '__main__':
    sys.exit(main())
