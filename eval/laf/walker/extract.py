"""Walker phase 1 — extract turns + candidates + labels (§20.2/§20.3).

Turn identity is (session_id, epoch, stop). Stop counters RESET on session
resume/compaction (604 colliding chain keys across 168 real sessions measured
2026-07-14), so (session, stop) alone is ambiguous. Epochs are derived per
session by sorting every stop-bearing trace row by timestamp and starting a
new epoch whenever the stop number regresses. A reset IS a context boundary —
the moment stack (§20.1) only looks within its own epoch, which implements
§20.3's "compaction seams respected" clause structurally.

Joins the trace legs structurally on (session, epoch, stop):
  Δ  s1 `additionalContext`   → outcomes_per_candidate (picked/dropped labels)
  O  s1 `recall`              → recall ts, stored query, candidate pool detail
  K  s1 `surface_selected`    → tool_trace (fetched_by / floored_by tiers)
  s0 user/assistant messages  → FULL turn texts (never the 500-char O query)
  s0 tool_result              → activity features
  s0 anchor_touched           → used-next-turn labels (surfacing-independent)

Within an epoch, duplicate Δ rows for one stop (hook retries — 391 burst keys
measured) dedupe to the LATEST row. Synthetic sessions (non-UUID ids, test
harnesses) are excluded. Every drop is COUNTED, never silent (§20.4).

Run:  ./dev python3 eval/laf/walker/extract.py
"""
import importlib.util
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from walker_db import fresh_walker, open_logs_ro, open_brain_ro, WALKER_DIR

REPO = Path(__file__).resolve().parents[3]
MANIFEST = WALKER_DIR / 'gold_manifest.json'

# trace_links is pure (json-only imports) — load by path to avoid pulling the
# servers package import chain into an offline script.
_spec = importlib.util.spec_from_file_location(
    'trace_links', REPO / 'servers' / 'scales' / 's1' / 'trace_links.py')
trace_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace_links)

FILE_TOOLS = {'Edit', 'Write', 'NotebookEdit'}
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

# streams a session timeline is built from (all stop-bearing)
S0_TYPES = ('user_message', 'assistant_message', 'tool_result', 'anchor_touched')


def stop_of(chain_id):
    tail = str(chain_id or '').rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else None


def norm(text):
    return re.sub(r'\s+', ' ', (text or '')).strip().lower()


def parse_ts(iso):
    try:
        return datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return None


def parse_candidate_line(line):
    """'shortid|title|score|type' — title may itself contain '|'."""
    parts = str(line).split('|')
    if len(parts) < 4:
        return None
    try:
        score = float(parts[-2])
    except ValueError:
        return None
    return {'short': parts[0], 'title': '|'.join(parts[1:-2]),
            'score': score, 'type': parts[-1]}


def jload(raw):
    try:
        return json.loads(raw or '{}')
    except (ValueError, TypeError):
        return {}


def assign_epochs(rows):
    """rows: [(created_at, stop, payload)] ONE session, any stream, unsorted.

    Sorts by timestamp; a new epoch starts whenever stop regresses. Returns
    [(epoch, stop, created_at, payload)]. Stop-less rows never reach here.
    """
    out = []
    epoch, prev_stop = 0, None
    for created, stop, payload in sorted(rows, key=lambda r: r[0]):
        if prev_stop is not None and stop < prev_stop:
            epoch += 1
        prev_stop = stop
        out.append((epoch, stop, created, payload))
    return out


def main():
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('unmatched'):
        print('FATAL: gold manifest has %d unmatched cues — walker refuses to build' %
              manifest['unmatched'])
        return 2
    gold_sessions = set(manifest['excluded_sessions'])

    logs = open_logs_ro()
    c = defaultdict(int)   # conservation counters

    # ── collect raw stop-bearing rows per session (one pass per scale) ──
    by_session = defaultdict(list)   # sess -> [(created, stop, (stream, meta...))]

    for sess, chain, created, meta_raw in logs.execute(
            "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
            "WHERE scale='s1' AND event_type='delta' AND ref_type='additionalContext'"):
        c['delta_rows'] += 1
        stop = stop_of(chain)
        if stop is None:
            c['delta_bad_chain'] += 1
            continue
        by_session[sess].append((created, stop, ('delta', meta_raw)))

    for sess, chain, created, meta_raw in logs.execute(
            "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
            "WHERE scale='s1' AND event_type='O' AND ref_type='recall'"):
        stop = stop_of(chain)
        if stop is not None:
            by_session[sess].append((created, stop, ('o', meta_raw)))

    for sess, chain, created, meta_raw in logs.execute(
            "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
            "WHERE scale='s1' AND event_type='K' AND ref_type='surface_selected'"):
        stop = stop_of(chain)
        if stop is not None:
            by_session[sess].append((created, stop, ('k', meta_raw)))

    for sess, chain, created, ref_type, meta_raw in logs.execute(
            "SELECT session_id, chain_id, created_at, ref_type, metadata FROM trace_events "
            "WHERE scale='s0' AND ref_type IN (%s)" % ','.join('?' * len(S0_TYPES)),
            S0_TYPES):
        stop = stop_of(chain)
        if stop is not None:
            by_session[sess].append((created, stop, (ref_type, meta_raw)))

    # ── session project (recent sessions only; older → NULL, expected) ──
    project_of = {}
    for sess, val in logs.execute(
            "SELECT session_id, value FROM session_state WHERE key='_session_context'"):
        proj = jload(val).get('project')
        if proj:
            project_of[sess] = proj
    logs.close()

    # ── node resolution map (brain.db, read-only) ───────────────────────
    braindb = open_brain_ro()
    prefix_map = defaultdict(list)
    node_times = {}
    for nid, created, updated in braindb.execute(
            "SELECT id, created_at, updated_at FROM nodes"):
        prefix_map[nid[:8]].append(nid)
        node_times[nid] = (created, updated)
    braindb.close()

    # ── per session: epochs → keyed stores → rows ───────────────────────
    walker = fresh_walker()
    turn_rows, cand_rows = [], []

    for sess, raw_rows in by_session.items():
        if sess in gold_sessions:
            c['sessions_gold_excluded_seen'] += 1
            continue
        if not UUID_RE.match(sess):
            c['sessions_synthetic'] += 1
            continue
        c['sessions_included'] += 1

        labels, o_rows, prov = {}, {}, {}
        s0_turn = defaultdict(dict)
        activity = defaultdict(lambda: {'tools': 0, 'files': 0})
        touched = defaultdict(set)

        for epoch, stop, created, (stream, meta_raw) in assign_epochs(raw_rows):
            key = (epoch, stop)
            if stream == 'delta':
                outcomes = jload(meta_raw).get('outcomes_per_candidate') or {}
                if not outcomes:
                    c['delta_empty_outcomes'] += 1
                    continue
                if key in labels:
                    c['delta_retry_deduped'] += 1   # keep LATEST (rows arrive ts-sorted)
                labels[key] = outcomes
            elif stream == 'o':
                meta = jload(meta_raw)
                if key in o_rows:
                    c['o_retry_deduped'] += 1
                o_rows[key] = {'ts': created, 'query': meta.get('query', ''),
                               'cands': meta.get('candidates') or []}
            elif stream == 'k':
                prov[key] = trace_links._tool_provenance(jload(meta_raw))
            elif stream == 'tool_result':
                activity[key]['tools'] += 1
                if jload(meta_raw).get('tool') in FILE_TOOLS:
                    activity[key]['files'] += 1
            elif stream == 'anchor_touched':
                meta = jload(meta_raw)
                for k in ('created', 'revised', 'recalled', 'endo'):
                    touched[key].update(meta.get(k) or [])
            elif stream == 'user_message':
                s0_turn[key]['op'] = jload(meta_raw).get('content', '')
                s0_turn[key]['op_ts'] = created
            elif stream == 'assistant_message':
                s0_turn[key].setdefault('anchor', jload(meta_raw).get('content', ''))

        # turns: every s0 turn in this session, per epoch
        disagree_keys = set()
        per_epoch = defaultdict(list)
        for (epoch, stop) in s0_turn:
            per_epoch[epoch].append(stop)
        for epoch, stops in per_epoch.items():
            stops.sort()
            prev_ts = None
            for i, stop in enumerate(stops):
                key = (epoch, stop)
                rec = s0_turn[key]
                o = o_rows.get(key, {})
                op_text = rec.get('op', '')
                flags = []
                if not op_text:
                    c['turn_no_op_text'] += 1
                    flags.append('no_op_text')
                q = o.get('query', '')
                if q and op_text:
                    a, b = norm(q)[:120], norm(op_text)[:120]
                    n = min(len(a), len(b))
                    if n >= 20 and a[:n] != b[:n]:
                        c['text_agreement_fail'] += 1
                        flags.append('text_disagree')
                        disagree_keys.add(key)
                t_now = parse_ts(rec.get('op_ts'))
                gap = (t_now - prev_ts).total_seconds() if (t_now and prev_ts) else None
                prev_ts = t_now
                act = activity[key]
                turn_rows.append((
                    sess, epoch, stop, o.get('ts') or rec.get('op_ts'),
                    op_text, rec.get('anchor', ''), q,
                    1 if key in labels else 0,
                    len(op_text), 1 if '```' in op_text else 0,
                    1 if '?' in op_text else 0,
                    act['tools'], act['files'], gap, i,
                    project_of.get(sess), json.dumps(flags)))

        # candidates: labeled turns only
        for (epoch, stop), outcomes in labels.items():
            key = (epoch, stop)
            o = o_rows.get(key)
            if o is None:
                c['label_missing_O'] += 1
                continue
            if key not in s0_turn:
                c['label_missing_s0'] += 1
            # text_disagree = the s0 text and the O query are DIFFERENT turns
            # (interrupted-turn misalignment, ~1%). A poisoned j=0 cue must not
            # feed the labeled set; the turn stays in `turns` as j>=1 context.
            # Residual-ledger item: realign via off-by-one recovery.
            if disagree_keys and key in disagree_keys:
                c['label_text_disagree_excluded'] += 1
                continue
            if not o['cands']:
                c['label_missing_candidates'] += 1     # the April gap class
                continue
            turn_ts = o['ts']
            fetched_by, floored_by = prov.get(key, ({}, {}))
            used1 = touched.get((epoch, stop + 1), set())
            used3 = (used1 | touched.get((epoch, stop + 2), set())
                     | touched.get((epoch, stop + 3), set()))
            seen_shorts = set()
            for rank, line in enumerate(o['cands']):
                parsed = parse_candidate_line(line)
                if parsed is None:
                    c['cand_unparseable'] += 1
                    continue
                short = parsed['short']
                seen_shorts.add(short)
                outcome = outcomes.get(short)
                if outcome is None:
                    c['cand_no_outcome'] += 1
                tier = 'picked' if outcome == 'selected' else 'pooled_dropped'
                cand_rows.append(_cand_row(
                    c, sess, epoch, stop, short, outcome, tier,
                    fetched_by.get(short), used1, used3, rank,
                    parsed['score'], turn_ts, prefix_map, node_times))
            for short, tool in floored_by.items():
                if short in seen_shorts:
                    continue
                cand_rows.append(_cand_row(
                    c, sess, epoch, stop, short, None, 'floored', tool,
                    used1, used3, None, None, turn_ts, prefix_map, node_times))
            c['labeled_turns_written'] += 1

    walker.executemany(
        "INSERT INTO turns (session_id, epoch, stop, ts, op_text, anchor_text,"
        " query_stored, labeled, op_len, has_code, has_question, tool_result_count,"
        " files_touched, gap_seconds, turns_since_start, project, flags)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", turn_rows)
    walker.executemany(
        "INSERT OR REPLACE INTO candidates (session_id, epoch, stop, cand_short,"
        " node_id, outcome, tier, fetched_by, used_next_1, used_next_3, rank_in_pool,"
        " pool_score, node_created_at, node_revised_after_turn, flags)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", cand_rows)
    c['turns_written'] = len(turn_rows)
    c['candidates_written'] = len(cand_rows)
    c['sessions_gold_excluded'] = len(gold_sessions)

    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('extract_' + k, str(v)) for k, v in sorted(c.items())])
    walker.commit()
    walker.close()

    print('extract phase — conservation counters:')
    for k in sorted(c):
        print('  %-28s %d' % (k, c[k]))
    return 0


def _cand_row(c, sess, epoch, stop, short, outcome, tier, fetched_tool,
              used1, used3, rank, score, turn_ts, prefix_map, node_times):
    flags = []
    full_ids = prefix_map.get(short, [])
    node_id = full_ids[0] if len(full_ids) == 1 else None
    if not full_ids:
        c['cand_unresolved'] += 1
        flags.append('unresolved')
    elif len(full_ids) > 1:
        c['cand_ambiguous'] += 1
        flags.append('ambiguous')
    created = updated = None
    revised_after = None
    if node_id:
        created, updated = node_times.get(node_id, (None, None))
        if updated and turn_ts:
            revised_after = 1 if updated > turn_ts else 0
    return (sess, epoch, stop, short, node_id, outcome, tier, fetched_tool,
            1 if short in used1 else 0, 1 if short in used3 else 0,
            rank, score, created, revised_after, json.dumps(flags))


if __name__ == '__main__':
    sys.exit(main())
