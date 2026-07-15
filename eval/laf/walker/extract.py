"""Walker phase 1 — extract micro-turns + candidates + labels (§20.2/§20.3).

TURN MODEL (v4 — v3 micro-turns + the 2026-07-14 untraced-taxonomy relabel,
brain node 9adc8127): a turn is anchored on a RECALL EVENT (the s1 O row), not
on the (session, stop) key. Stop counters collide twice over:

  • RESETS — resume/compaction restarts the counter (604 colliding keys / 168
    sessions measured). Fixed by EPOCHS: per session, sort every stop-bearing
    row by timestamp and start a new epoch when the stop regresses. The moment
    stack never crosses an epoch boundary (§20.3 compaction-seams clause).
  • SAME-STOP COLLISIONS — steering messages, Esc-interrupts, and injected
    task-notifications all land before the turn's Stop fires, so several
    recalls share one stop. Each O row becomes its own MICRO-TURN, ordered by
    ts (`seq`). Flags (taxonomy verified 2026-07-14, brain node 9adc8127):
      - `untraced_legacy`: O row with NO s0 user_message at its stop. Almost
        entirely pre-2026-06-08, when user_message was written at Stop
        (94b4642 moved it to prompt-arrival — the class ends there).
        op_text = the O query, ≤500 chars.
      - `superseded`: this turn shares its stop with others and is NOT the
        stop's SURVIVOR — the turn holding the recorded response (else the
        stop's last turn). Steering / interrupt / notification collisions;
        live signal, era-independent, ~7-12% of turns every month.
        no_recall bookkeeping turns are never flagged.
      - `text_disagree`: an s0 existed at the stop but text agreement failed;
        paired STRUCTURALLY (latest unpaired O) so the moment stack keeps the
        full operator text, but excluded from labels (op/query mismatch).
      - `no_recall`: s0 turn with no O row at all — register_only short
        answers, task-notification skips (post-2026-07-03), rare hook misses.
    Nothing is key-deduped away — v2's keep-latest dedup silently discarded
    every same-stop first recall (~800 O rows).
    The Stop's assistant_message attaches to the LAST s0 of its stop: the one
    combined response answers the final steering message, not the first.

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

from walker_db import (fresh_walker, open_logs_ro, open_brain_ro, WALKER_DIR,
                       gold_source_hash, EXTRACT_VERSION)

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


stop_of = trace_links._stop_of      # reuse the canonical join-key parser (review F7)
                                    # — one definition, so a chain-shape change
                                    # can't fix the live join and miss the walker's


def norm(text):
    return re.sub(r'\s+', ' ', (text or '')).strip().lower()


def texts_agree(a, b, floor=40, span=120):
    """Do two normalized texts denote the same operator message? Long prefix
    agreement (>= floor chars), OR — for texts shorter than floor — full
    equality. The short-text branch is STRICTER than raw prefix containment on
    purpose (review F5): otherwise a 2-char 'ok' would pair to any O query
    starting 'ok...', mis-attaching that recall's labels. (Deliberately tighter
    than the gold manifest's cue matcher, which pairs a short cue to a longer
    trace prefix — a different job.)"""
    a, b = norm(a)[:span], norm(b)[:span]
    n = min(len(a), len(b))
    if n == 0:
        return False
    return a[:n] == b[:n] and (n >= floor or len(a) == len(b))


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
    for epoch, stop, created, (stream, meta_raw, rid) in assign_epochs(raw_rows):
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
            ep['s0'].append({'ts': created, 'stop': stop, 'trace_id': rid,
                             'op': jload(meta_raw).get('content', ''),
                             'anchor': None, 'anchor_trace_id': None})
        elif stream == 'assistant_message':
            # attach to the LAST s0 of the stop: with steering messages the
            # single Stop-time response answers the final prompt, not the
            # first (all s0 rows of the stop precede this event by ts)
            for rec in reversed(ep['s0']):
                if rec['stop'] == stop and rec['anchor'] is None:
                    rec['anchor'] = jload(meta_raw).get('content', '')
                    rec['anchor_trace_id'] = rid
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

        # s0 user_message → agreeing O at same stop. Two passes so text
        # agreement claims its O before any structural fallback grabs it.
        disagree_pending, standalone = [], []
        for rec in ep['s0']:
            cands = _unpaired_os_at_stop(ep['o'], rec['stop'])
            hit = None
            for o in reversed(cands):                     # prefer latest
                if texts_agree(o['query'], rec['op']):
                    hit = o
                    break
            if hit is not None:
                hit['s0'] = rec
            else:
                disagree_pending.append(rec)
        for rec in disagree_pending:
            # structural fallback: an s0 and an unpaired O at the same stop are
            # the same turn even when texts disagree (command expansion, prompt
            # rewriting). Pair the latest unpaired O so the moment stack keeps
            # the full operator text + response; flagged text_disagree below
            # and excluded from labels.
            cands = _unpaired_os_at_stop(ep['o'], rec['stop'])
            if cands:
                cands[-1]['s0'] = rec
                cands[-1]['disagree'] = True
                c['s0_paired_text_disagree'] += 1
            else:
                if any(o['stop'] == rec['stop'] for o in ep['o']):
                    c['s0_disagree_unpairable'] += 1      # every O at this stop already claimed
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
                    flags.append('untraced_legacy')       # prompt s0 never recorded
                    c['turns_untraced_legacy'] += 1       # (pre-06-08 Stop-time write)
                elif obj.get('disagree'):
                    flags.append('text_disagree')
                op_text = rec['op'] if rec else obj['query']
                turns.append({
                    'ts': obj['ts'], 'stop': obj['stop'], 'o': obj,
                    'op': op_text, 'anchor': (rec or {}).get('anchor') or '',
                    'query': obj['query'], 'flags': flags,
                    'op_tid': (rec or {}).get('trace_id'),
                    'anchor_tid': (rec or {}).get('anchor_trace_id'),
                    'msg_ts': (rec or {}).get('ts') or obj['ts']})
            else:
                c['turns_no_recall'] += 1
                turns.append({
                    'ts': obj['ts'], 'stop': obj['stop'], 'o': None,
                    'op': obj['op'], 'anchor': obj.get('anchor') or '',
                    'query': '', 'flags': ['no_recall'],
                    'op_tid': obj.get('trace_id'),
                    'anchor_tid': obj.get('anchor_trace_id'),
                    'msg_ts': obj['ts']})

        # superseded: at a multi-turn stop, every turn except the SURVIVOR —
        # the turn holding the stop's one recorded response (the anchor
        # attaches to the last s0 of the stop; when no response was recorded
        # the stop's last turn survives). Keyed on the anchor-holder, not turn
        # order: with interrupts, prompt-ts order and recall-ts order can
        # cross, and the survivor must be the turn that owns the response.
        # no_recall bookkeeping turns (register_only / notification skips)
        # are never flagged — they aren't steering/interrupt supersession.
        idxs_of_stop = defaultdict(list)
        for idx, t in enumerate(turns):
            idxs_of_stop[t['stop']].append(idx)
        for idxs in idxs_of_stop.values():
            if len(idxs) < 2:
                continue
            holders = [i for i in idxs if turns[i]['anchor']]
            survivor = holders[-1] if holders else idxs[-1]
            for i in idxs:
                if i != survivor and 'no_recall' not in turns[i]['flags']:
                    turns[i]['flags'].append('superseded')
                    c['turns_superseded'] += 1

        # rows + labels, seq-ordered
        prev_ts = None
        for seq, t in enumerate(turns):
            act = ep['act'][t['stop']] if t['o'] is None or t['o']['s0'] else {'tools': 0, 'files': 0}
            t_now = parse_ts(t['msg_ts'])
            gap = (t_now - prev_ts).total_seconds() if (t_now and prev_ts) else None
            prev_ts = t_now
            labeled = bool(t['o'] and t['o']['outcomes'] and t['o']['cands'])
            if labeled and 'text_disagree' in t['flags']:
                labeled = False                           # op/query mismatch — keep as
                c['label_excluded_text_disagree'] += 1    # context, not as a label row
            turn_rows.append((
                sess, epoch, seq, t['stop'], t['ts'], t['op'], t['anchor'],
                t['query'], t['op_tid'], t['anchor_tid'],
                1 if labeled else 0,
                len(t['op']), 1 if '```' in t['op'] else 0,
                1 if '?' in t['op'] else 0,
                act['tools'], act['files'], gap,
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
                    if nxt['stop'] == t['stop']:
                        continue          # same-stop successor (superseded): its
                        # touched-set is the shared per-stop accumulator, which
                        # already holds THIS turn's own usage — counting it
                        # self-leaks the label (review F2). anchor_touched is
                        # collected per stop, not per micro-turn.
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


def _unpaired_os_at_stop(o_events, stop):
    """The one pairing predicate both the text-agreement pass and the
    structural fallback use — ts-ordered Os at this stop not yet claimed."""
    return [o for o in o_events if o['stop'] == stop and o['s0'] is None]


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
    # Manifest freshness (wrong-science hardening): the exclusion set was built
    # from specific gold files. If the gold corpus changed since (gold-growth
    # is planned), building with the stale manifest LEAKS the new cues'
    # sessions into the training substrate — refuse instead.
    stale = {name: h for name, h in (manifest.get('source_hashes') or {}).items()
             if gold_source_hash(name) != h}
    if not manifest.get('source_hashes'):
        print('FATAL: manifest predates source-hash stamping — regenerate: '
              './dev python3 eval/laf/walker/gold_manifest.py')
        return 2
    if stale:
        print('FATAL: gold corpus changed since the manifest was built (%s) — '
              'regenerate gold_manifest.py before extracting, or the new '
              'cues\' sessions leak into training.' % ', '.join(sorted(stale)))
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
            by_session[sess].append((created, stop, (stream, meta_raw, None)))

    for rid, sess, chain, created, ref_type, meta_raw in logs.execute(
            "SELECT id, session_id, chain_id, created_at, ref_type, metadata FROM trace_events "
            "WHERE scale='s0' AND ref_type IN (%s)" % ','.join('?' * len(S0_TYPES)),
            S0_TYPES):
        stop = stop_of(chain)
        if stop is not None:
            by_session[sess].append((created, stop, (ref_type, meta_raw, rid)))

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
        " query_stored, op_trace_id, anchor_trace_id, labeled, op_len, has_code,"
        " has_question, tool_result_count, files_touched, gap_seconds,"
        " project, flags)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", all_turns)
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
        [('extract_' + k, str(v)) for k, v in sorted(c.items())]
        + [('extract_version', EXTRACT_VERSION),
           ('manifest_source_hashes', json.dumps(manifest['source_hashes']))])
    walker.commit()
    walker.close()

    print('extract phase (v4 micro-turns) — conservation counters:')
    for k in sorted(c):
        print('  %-28s %d' % (k, c[k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
