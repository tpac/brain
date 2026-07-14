"""Walker phase 1 — extract micro-turns + candidates + labels (§20.2/§20.3).

TURN MODEL (v3, Tom 2026-07-14: "can't it stay in the normal conversation
flow?"): a turn is anchored on a RECALL EVENT (the s1 O row), not on the
(session, stop) key. Stop counters collide twice over:

  • RESETS — resume/compaction restarts the counter (604 colliding keys / 168
    sessions measured). Fixed by EPOCHS: per session, sort every stop-bearing
    row by timestamp and start a new epoch when the stop regresses. The moment
    stack never crosses an epoch boundary (§20.3 compaction-seams clause).
  • INTERRUPTS — operator interrupts and re-prompts; both recalls share one
    stop. The interrupted prompt IS part of the conversation flow: its O row
    preserves the prompt text, its Δ row the judge labels. Each O row becomes
    its own MICRO-TURN, ordered by ts (`seq`); an O row with no agreeing s0
    text is an interrupted turn (op_text = the O query, ≤500 chars); an s0
    turn with no agreeing O row is a no-recall turn (hook timeout/failure).
    Nothing is key-deduped away — v2's keep-latest dedup silently discarded
    every interrupt-then-reprompt first recall (~800 O rows).

Pairing, all within (session, epoch), all ts-ordered:
  Δ / K  → the latest O row with same stop and O.ts <= row.ts (O,K,Δ are
           written in one append_batch in that order — verified surface.py:777)
  s0 user_message → the agreeing O row at the same stop (prefix agreement,
           short texts must match whole); no agreement → standalone s0 turn
  s0 assistant/tool_result/anchor_touched → the turn holding that stop's
           user_message
  used_next_k → union of anchor_touched over the next k turns BY SEQ

Synthetic sessions (non-UUID ids) are excluded. Every drop is COUNTED (§20.4).

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
S0_TYPES = ('user_message', 'assistant_message', 'tool_result', 'anchor_touched')
USED_NEXT_WINDOWS = (1, 3)


def stop_of(chain_id):
    tail = str(chain_id or '').rsplit('-', 1)[-1]
    return int(tail) if tail.isdigit() else None


def norm(text):
    return re.sub(r'\s+', ' ', (text or '')).strip().lower()


def texts_agree(a, b, floor=40, span=120):
    """Prefix agreement between two normalized texts; short texts must match
    their entire length (same rule as the gold manifest)."""
    a, b = norm(a)[:span], norm(b)[:span]
    n = min(len(a), len(b))
    if n == 0:
        return False
    return a[:n] == b[:n] and n >= min(floor, len(a), len(b))


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
    """rows: [(created_at, stop, payload)] ONE session, unsorted. Sorts by ts;
    a new epoch starts whenever stop regresses."""
    out = []
    epoch, prev_stop = 0, None
    for created, stop, payload in sorted(rows, key=lambda r: r[0]):
        if prev_stop is not None and stop < prev_stop:
            epoch += 1
        prev_stop = stop
        out.append((epoch, stop, created, payload))
    return out


def process_session(sess, raw_rows, project, prefix_map, node_times, c):
    """One session → (turn_rows, cand_rows). Pure over its inputs."""
    # per-epoch event streams, ts-ordered by construction
    epochs = defaultdict(lambda: {'o': [], 's0': [], 'delta': [], 'k': [],
                                  'act': defaultdict(lambda: {'tools': 0, 'files': 0}),
                                  'touched': defaultdict(set)})
    for epoch, stop, created, (stream, meta_raw) in assign_epochs(raw_rows):
        ep = epochs[epoch]
        if stream == 'o':
            meta = jload(meta_raw)
            ep['o'].append({'ts': created, 'stop': stop,
                            'query': meta.get('query', ''),
                            'cands': meta.get('candidates') or [],
                            'outcomes': None, 'prov': ({}, {}),
                            's0': None})
        elif stream == 'delta':
            ep['delta'].append((created, stop, jload(meta_raw).get('outcomes_per_candidate') or {}))
        elif stream == 'k':
            ep['k'].append((created, stop, trace_links._tool_provenance(jload(meta_raw))))
        elif stream == 'tool_result':
            ep['act'][stop]['tools'] += 1
            if jload(meta_raw).get('tool') in FILE_TOOLS:
                ep['act'][stop]['files'] += 1
        elif stream == 'anchor_touched':
            meta = jload(meta_raw)
            for k in ('created', 'revised', 'recalled', 'endo'):
                ep['touched'][stop].update(meta.get(k) or [])
        elif stream == 'user_message':
            ep['s0'].append({'ts': created, 'stop': stop,
                             'op': jload(meta_raw).get('content', ''), 'anchor': None})
        elif stream == 'assistant_message':
            # first assistant message of a stop = the turn's response
            for rec in ep['s0']:
                if rec['stop'] == stop and rec['anchor'] is None:
                    rec['anchor'] = jload(meta_raw).get('content', '')
                    break

    turn_rows, cand_rows = [], []
    for epoch, ep in epochs.items():
        # Δ/K → latest O with same stop, O.ts <= row.ts
        for created, stop, payload in ep['delta']:
            tgt = _latest_o(ep['o'], stop, created)
            if tgt is None:
                if payload:
                    c['delta_unpaired'] += 1
                continue
            if not payload:
                c['delta_empty_outcomes'] += 1
                continue
            if tgt['outcomes'] is not None:
                c['delta_double_pair'] += 1
            tgt['outcomes'] = payload
        for created, stop, prov in ep['k']:
            tgt = _latest_o(ep['o'], stop, created)
            if tgt is not None:
                tgt['prov'] = prov

        # s0 user_message → agreeing O at same stop
        standalone = []
        for rec in ep['s0']:
            cands = [o for o in ep['o'] if o['stop'] == rec['stop'] and o['s0'] is None]
            hit = None
            for o in reversed(cands):                     # prefer latest
                if texts_agree(o['query'], rec['op']):
                    hit = o
                    break
            if hit is not None:
                hit['s0'] = rec
            else:
                if cands:
                    c['s0_no_agreeing_o'] += 1            # replacement whose recall failed AND text differs
                standalone.append(rec)

        # micro-turn sequence: O-anchored turns + standalone s0, by ts
        seq_events = [('o', o['ts'], o) for o in ep['o']]
        seq_events += [('s0', rec['ts'], rec) for rec in standalone]
        seq_events.sort(key=lambda e: e[1])

        turns = []
        for kind, ts, obj in seq_events:
            if kind == 'o':
                rec = obj['s0']
                flags = []
                if rec is None:
                    flags.append('interrupted')           # prompt s0 never recorded
                    c['turns_interrupted'] += 1
                op_text = rec['op'] if rec else obj['query']
                turns.append({
                    'ts': obj['ts'], 'stop': obj['stop'], 'o': obj,
                    'op': op_text, 'anchor': (rec or {}).get('anchor') or '',
                    'query': obj['query'], 'flags': flags,
                    'msg_ts': (rec or {}).get('ts') or obj['ts']})
            else:
                c['turns_no_recall'] += 1
                turns.append({
                    'ts': obj['ts'], 'stop': obj['stop'], 'o': None,
                    'op': obj['op'], 'anchor': obj.get('anchor') or '',
                    'query': '', 'flags': ['no_recall'], 'msg_ts': obj['ts']})

        # rows + labels, seq-ordered
        prev_ts = None
        for seq, t in enumerate(turns):
            act = ep['act'][t['stop']] if t['o'] is None or t['o']['s0'] else {'tools': 0, 'files': 0}
            t_now = parse_ts(t['msg_ts'])
            gap = (t_now - prev_ts).total_seconds() if (t_now and prev_ts) else None
            prev_ts = t_now
            labeled = bool(t['o'] and t['o']['outcomes'] and t['o']['cands'])
            turn_rows.append((
                sess, epoch, seq, t['stop'], t['ts'], t['op'], t['anchor'],
                t['query'], 1 if labeled else 0,
                len(t['op']), 1 if '```' in t['op'] else 0,
                1 if '?' in t['op'] else 0,
                act['tools'], act['files'], gap, seq,
                project, json.dumps(t['flags'])))
            if not labeled:
                if t['o'] and t['o']['outcomes'] and not t['o']['cands']:
                    c['label_missing_candidates'] += 1    # the April gap class
                continue
            c['labeled_turns_written'] += 1
            used = {}
            for w in USED_NEXT_WINDOWS:
                ids = set()
                for nxt in turns[seq + 1: seq + 1 + w]:
                    ids |= ep['touched'].get(nxt['stop'], set())
                used[w] = ids
            o = t['o']
            fetched_by, floored_by = o['prov']
            turn_ts = o['ts']
            seen = set()
            for rank, line in enumerate(o['cands']):
                parsed = parse_candidate_line(line)
                if parsed is None:
                    c['cand_unparseable'] += 1
                    continue
                seen.add(parsed['short'])
                outcome = o['outcomes'].get(parsed['short'])
                if outcome is None:
                    c['cand_no_outcome'] += 1
                tier = 'picked' if outcome == 'selected' else 'pooled_dropped'
                cand_rows.append(_cand_row(
                    c, sess, epoch, seq, parsed['short'], outcome, tier,
                    fetched_by.get(parsed['short']), used, rank,
                    parsed['score'], turn_ts, prefix_map, node_times))
            for short, tool in floored_by.items():
                if short not in seen:
                    cand_rows.append(_cand_row(
                        c, sess, epoch, seq, short, None, 'floored', tool,
                        used, None, None, turn_ts, prefix_map, node_times))
    return turn_rows, cand_rows


def _latest_o(o_events, stop, created):
    tgt = None
    for o in o_events:
        if o['stop'] == stop and o['ts'] <= created:
            tgt = o
    return tgt


def _cand_row(c, sess, epoch, seq, short, outcome, tier, fetched_tool,
              used, rank, score, turn_ts, prefix_map, node_times):
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
    u1 = 1 if used and short in used[1] else 0
    u3 = 1 if used and short in used[3] else 0
    return (sess, epoch, seq, short, node_id, outcome, tier, fetched_tool,
            u1, u3, rank, score, created, revised_after, json.dumps(flags))


def main():
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get('unmatched'):
        print('FATAL: gold manifest has %d unmatched cues — walker refuses to build' %
              manifest['unmatched'])
        return 2
    gold_sessions = set(manifest['excluded_sessions'])

    logs = open_logs_ro()
    c = defaultdict(int)
    by_session = defaultdict(list)

    for stream, sql in (
            ('delta', "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
                      "WHERE scale='s1' AND event_type='delta' AND ref_type='additionalContext'"),
            ('o', "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
                  "WHERE scale='s1' AND event_type='O' AND ref_type='recall'"),
            ('k', "SELECT session_id, chain_id, created_at, metadata FROM trace_events "
                  "WHERE scale='s1' AND event_type='K' AND ref_type='surface_selected'")):
        for sess, chain, created, meta_raw in logs.execute(sql):
            if stream == 'delta':
                c['delta_rows'] += 1
            stop = stop_of(chain)
            if stop is None:
                c[stream + '_bad_chain'] += 1
                continue
            by_session[sess].append((created, stop, (stream, meta_raw)))

    for sess, chain, created, ref_type, meta_raw in logs.execute(
            "SELECT session_id, chain_id, created_at, ref_type, metadata FROM trace_events "
            "WHERE scale='s0' AND ref_type IN (%s)" % ','.join('?' * len(S0_TYPES)),
            S0_TYPES):
        stop = stop_of(chain)
        if stop is not None:
            by_session[sess].append((created, stop, (ref_type, meta_raw)))

    project_of = {}
    for sess, val in logs.execute(
            "SELECT session_id, value FROM session_state WHERE key='_session_context'"):
        proj = jload(val).get('project')
        if proj:
            project_of[sess] = proj
    logs.close()

    braindb = open_brain_ro()
    prefix_map = defaultdict(list)
    node_times = {}
    for nid, created, updated in braindb.execute(
            "SELECT id, created_at, updated_at FROM nodes"):
        prefix_map[nid[:8]].append(nid)
        node_times[nid] = (created, updated)
    braindb.close()

    walker = fresh_walker()
    all_turns, all_cands = [], []
    for sess, raw_rows in by_session.items():
        if sess in gold_sessions:
            c['sessions_gold_excluded_seen'] += 1
            continue
        if not UUID_RE.match(sess):
            c['sessions_synthetic'] += 1
            continue
        c['sessions_included'] += 1
        t_rows, c_rows = process_session(
            sess, raw_rows, project_of.get(sess), prefix_map, node_times, c)
        all_turns.extend(t_rows)
        all_cands.extend(c_rows)

    walker.executemany(
        "INSERT INTO turns (session_id, epoch, seq, stop, ts, op_text, anchor_text,"
        " query_stored, labeled, op_len, has_code, has_question, tool_result_count,"
        " files_touched, gap_seconds, turns_since_start, project, flags)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", all_turns)
    walker.executemany(
        "INSERT OR REPLACE INTO candidates (session_id, epoch, seq, cand_short,"
        " node_id, outcome, tier, fetched_by, used_next_1, used_next_3, rank_in_pool,"
        " pool_score, node_created_at, node_revised_after_turn, flags)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", all_cands)
    c['turns_written'] = len(all_turns)
    c['candidates_written'] = len(all_cands)
    c['sessions_gold_excluded'] = len(gold_sessions)

    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('extract_' + k, str(v)) for k, v in sorted(c.items())])
    walker.commit()
    walker.close()

    print('extract phase (v3 micro-turns) — conservation counters:')
    for k in sorted(c):
        print('  %-28s %d' % (k, c[k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
