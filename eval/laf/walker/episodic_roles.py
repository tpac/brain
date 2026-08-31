"""Walker phase 5 — per-(turn, message-offset) episodic lanes (§20.5 grid).

The Q1 grid runs episodic (pick/enc) through the IDENTICAL moment-stack grid
as the content lanes (H2 lock: no special-casing) — which needs, per labeled
turn t, candidate n, offset j: the episodic activation the engine would give
n if the query were t's j-th previous message, evaluated AS-OF t. This is
"matvec + precomputed as-of role map":

  matvec      message-j vector against the engine's trace matrix → top-15
              similar moments (±1-stop window, max-sim dedup) — the exact
              _episodic_vectors selection semantics, as-of masked
  role map    per (session, stop): picked/dropped shorts with their surface
              row's created_at; encode runs as an ORDERED (run_stop,
              created_at, ids) list. As-of filtering happens at LOOKUP
              (created_at ≤ turn ts), because pre-joining would bake in
              encode runs the turn's as_of should not see (runs lag their
              turns); the owning run as-of = first VISIBLE run with
              run_stop > stop — nodes_for_traces' STRICT join, restricted
              to runs created ≤ as_of.

SELF-CHECK (wrong-science discipline — prove the shortcut IS the engine):
for a random sample of (turn, j) pairs, the table values are recomputed via
the production engine's _episodic_vectors(as_of=, trace_mask=) directly and
must agree to float tolerance; the result is stamped into build_meta and a
mismatch fails the build. No sweep may read this table without the stamp.

Columns per (turn, node, j): pick_op/enc_op (query = j-th previous operator
message; j=0 uses q_vec exactly like scores.py) and pick_anchor/enc_anchor
(query = j-th previous attached assistant message, for the texts=op+anchor
arm; NULL where that turn has no anchor).

Run:  ./dev python3 eval/laf/walker/episodic_roles.py [--rebuild]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import (open_walker, EXTRACT_VERSION, EMBED_VERSION,
                       lanes_version, WALKER_DIR)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (LafV1Engine, MAXSIM_VIEWS, _unit,   # noqa: E402
                                DEFAULT_CONFIG, role_rows)
from servers.scales.s1.trace_links import (_stop_of, _surface_ids,  # noqa: E402
                                           _delta_ids, _candidate_outcomes,
                                           GATHER_STREAMS)

K_MAX = 8
TOP_MOMENTS = int(DEFAULT_CONFIG['top_moments'])      # engine parity
WINDOW = int(DEFAULT_CONFIG['window_turns'])          # ±1, engine parity
SELF_CHECK_SAMPLES = 60
TOL = 1e-6
EPISODIC_VERSION = 'v1-eng-parity-top%d-w%d' % (TOP_MOMENTS, WINDOW)


def gate_provenance(walker):
    stamps = dict(walker.execute(
        "SELECT key, value FROM build_meta WHERE key IN "
        "('extract_version','embed_version','scores_lanes_version')"))
    expect = {'extract_version': EXTRACT_VERSION,
              'embed_version': EMBED_VERSION,
              'scores_lanes_version': lanes_version(MAXSIM_VIEWS)}
    bad = {k for k, v in expect.items() if stamps.get(k) != v}
    if bad:
        raise SystemExit('episodic_roles: stale walker artifact (%s) — '
                         'rebuild first, never bypass.' % ', '.join(bad))


def build_role_map(brain):
    """Per-session role records WITH timestamps, for lookup-time as-of.

    Returns {session_id: {'surf': {stop: [(created, [short_ids])]},
                          'pool': {stop: [(created, [short_ids], explicit)]},
                          'runs': [(run_stop, created, [full_ids])] ASC}}
    Streams pulled through the same query_traces door gather() uses
    (surface / encode / recall), uncapped per session.
    """
    out = {}
    sessions = [r[0] for r in brain._trace_dal.conn.execute(
        "SELECT DISTINCT session_id FROM trace_events WHERE scale='s1'")]
    for sess in sessions:
        rec = {'surf': defaultdict(list), 'pool': defaultdict(list),
               'runs': []}
        for name in ('surface', 'encode', 'recall'):
            ref_type, scale = GATHER_STREAMS[name]
            events = brain.query_traces(
                ref_type=ref_type, scale=scale, session_id=sess,
                hours=None, limit=100000).get('events', [])
            for t in events:
                stop = _stop_of(t.get('chain_id'))
                if stop is None:
                    continue
                created = t.get('created_at') or ''
                if name == 'surface':
                    ids = _surface_ids(t.get('ref_id'))
                    if ids:
                        rec['surf'][stop].append((created, ids))
                elif name == 'encode':
                    ids = _delta_ids(t.get('metadata'), 'created', 'revised')
                    rec['runs'].append((stop, created, ids))
                else:
                    cands, dropped = _candidate_outcomes(t.get('metadata')
                                                         or {})
                    if cands:
                        rec['pool'][stop].append(
                            (created, cands, dropped))
        rec['runs'].sort(key=lambda r: r[0])
        out[sess] = rec
    return out


def roles_at(rec, stop, as_of):
    """(picked, encoded, dropped) sets for ONE stop as-of — mirrors
    nodes_for_traces: surfaced from visible surface rows; owning run =
    first VISIBLE run with run_stop > stop (STRICT); dropped from visible
    pool rows (explicit verdict wins), minus picked at moment level (the
    caller unions the ±window first, per roles_for_moments)."""
    picked = set()
    for created, ids in rec['surf'].get(stop, []):
        if created <= as_of:
            picked.update(ids)
    encoded = set()
    for run_stop, created, ids in rec['runs']:
        if run_stop > stop and created <= as_of:
            encoded.update(ids)
            break
    dropped = set()
    pool_rows = [(c, cands, expl) for c, cands, expl
                 in rec['pool'].get(stop, []) if c <= as_of]
    explicit = [expl for _, _, expl in pool_rows if expl is not None]
    if explicit:
        for e in explicit:
            dropped.update(e)
    else:
        for _, cands, _ in pool_rows:
            dropped.update(cands)
    return picked, encoded, dropped


def episodic_from_sims(eng, brain, role_map, sims, as_of, trace_created):
    """{node_row: (pick, enc)} — the engine's _episodic_vectors semantics
    over the precomputed role map (top-K moment scan, ±WINDOW union,
    picked-wins, max score per node). `sims` is the trace-matrix matvec for
    ONE query vector (the caller batches those); selection is argpartition-
    then-sort — identical top slice to the engine's full argsort, faster."""
    sims = np.where(trace_created <= as_of, sims, -np.inf)
    k = min(TOP_MOMENTS * 3, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    moments = {}
    for i in top:
        chain = eng._tr_meta[i][0] or ''
        stop = _stop_of(chain)
        parts = chain.split('-')
        sess = eng._tr_meta[i][1]
        if stop is None or not sess or len(parts) < 3:
            continue
        key = (sess, parts[1], stop)
        s = float(sims[i])
        if s > moments.get(key, 0.0):
            moments[key] = s
        if len(moments) >= TOP_MOMENTS:
            break
    out = {}
    for (sess, _short, stop), s in moments.items():
        rec = role_map.get(sess)
        if rec is None:
            continue
        picked, encoded, dropped = set(), set(), set()
        for ws in range(max(stop - WINDOW, 0), stop + WINDOW + 1):
            p, e, d = roles_at(rec, ws, as_of)
            picked |= p
            encoded |= e
            dropped |= d
        dropped -= picked
        # Survivor-credit through the PRODUCTION helper — the engine credits an
        # absorbed id to its live survivor's row, so a bare _resolve here would
        # diverge by a whole moment score and trip the self-check below.
        rows, _ = role_rows(brain, picked | encoded, eng._resolve)
        for nid in picked:
            r = rows.get(nid)
            if r is not None and s > out.get(r, (0.0, 0.0))[0]:
                out[r] = (s, out.get(r, (0.0, 0.0))[1])
        for nid in encoded:
            r = rows.get(nid)
            if r is not None and s > out.get(r, (0.0, 0.0))[1]:
                out[r] = (out.get(r, (0.0, 0.0))[0], s)
    return out


def main():
    rebuild = '--rebuild' in sys.argv
    walker = open_walker()
    gate_provenance(walker)
    have = walker.execute("SELECT value FROM build_meta WHERE "
                          "key='episodic_roles_version'").fetchone()
    if have and have[0] == EPISODIC_VERSION and not rebuild:
        print('cand_turn_episodic current (%s) — nothing to do' % have[0])
        return 0
    if have and have[0] != EPISODIC_VERSION and not rebuild:
        raise SystemExit('episodic stamped %s, code is %s — rerun with '
                         '--rebuild' % (have[0], EPISODIC_VERSION))
    walker.execute('DROP TABLE IF EXISTS cand_turn_episodic')
    walker.execute(
        'CREATE TABLE cand_turn_episodic ('
        ' session_id TEXT NOT NULL, epoch INTEGER NOT NULL,'
        ' seq INTEGER NOT NULL, node_id TEXT NOT NULL, j INTEGER NOT NULL,'
        ' pick_op REAL, enc_op REAL, pick_anchor REAL, enc_anchor REAL,'
        ' PRIMARY KEY (session_id, epoch, seq, node_id, j))')
    walker.commit()

    # turn stacks (scores.py shape: j=0 op source is q_vec)
    turns = defaultdict(dict)
    for sess, epoch, seq, opv, av, qv in walker.execute(
            "SELECT session_id, epoch, seq, op_vec, anchor_vec, q_vec "
            "FROM turns"):
        turns[(sess, epoch)][seq] = (_unit(opv) if opv else None,
                                     _unit(av) if av else None,
                                     _unit(qv) if qv else None)
    cand_by_turn = defaultdict(list)
    for sess, epoch, seq, nid in walker.execute(
            "SELECT c.session_id, c.epoch, c.seq, c.node_id FROM candidates c"
            " JOIN turns t ON t.session_id=c.session_id AND t.epoch=c.epoch"
            "  AND t.seq=c.seq WHERE t.labeled=1 AND c.node_id IS NOT NULL"):
        cand_by_turn[(sess, epoch, seq)].append(nid)
    turn_ts = {(s, e, q): ts for s, e, q, ts in walker.execute(
        "SELECT session_id, epoch, seq, ts FROM turns WHERE labeled=1")}

    from tests.isolated_brain import IsolatedBrain
    c = defaultdict(int)
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)     # one matrix, batched matvecs
        role_map = build_role_map(env.brain)
        c['role_map_sessions'] = len(role_map)

        ins = ('INSERT OR REPLACE INTO cand_turn_episodic (session_id, epoch,'
               ' seq, node_id, j, pick_op, enc_op, pick_anchor, enc_anchor)'
               ' VALUES (?,?,?,?,?,?,?,?,?)')
        buf = []
        check_pool = []            # (key, j, vec, kind) for the self-check
        for key, cands in sorted(cand_by_turn.items()):
            sess, epoch, seq = key
            ts = turn_ts.get(key)
            if ts is None:
                c['turns_missing_ts'] += 1
                continue
            rows_idx = [(nid, eng._idx.get(nid)) for nid in cands]
            epoch_turns = turns[(sess, epoch)]
            # collect the turn's message vectors, ONE batched matmul
            queries = []                     # (j, kind, vec)
            for j in range(0, K_MAX + 1):
                src = epoch_turns.get(seq - j)
                if src is None:
                    break
                op_vec, anchor_vec, q_vec = src
                op_j = q_vec if j == 0 else op_vec
                for kind, vec in (('op', op_j), ('anchor', anchor_vec)):
                    if vec is None:
                        c['j_missing_%s_vec' % kind] += 1
                    else:
                        queries.append((j, kind, vec))
            j_max = max((j for j, _, _ in queries), default=-1)
            if not queries:
                continue
            sims_all = trace_mat @ np.stack([v for _, _, v in queries]).T
            vals = {}                        # (j, kind) → roles dict
            for col, (j, kind, vec) in enumerate(queries):
                vals[(j, kind)] = episodic_from_sims(
                    eng, env.brain, role_map, sims_all[:, col], ts,
                    trace_created)
                check_pool.append((key, j, kind, vec, ts))
            for j in range(0, j_max + 1):
                for nid, r in rows_idx:
                    def cell(kind, slot):
                        v = vals.get((j, kind))
                        if v is None:
                            return None
                        return v.get(r, (0.0, 0.0))[slot] if r is not None \
                            else 0.0
                    buf.append((sess, epoch, seq, nid, j,
                                cell('op', 0), cell('op', 1),
                                cell('anchor', 0), cell('anchor', 1)))
            c['turns_done'] += 1
            if len(buf) >= 20000:
                walker.executemany(ins, buf)
                walker.commit()
                c['rows_written'] += len(buf)
                buf = []
        if buf:
            walker.executemany(ins, buf)
            c['rows_written'] += len(buf)

        # SELF-CHECK: shortcut ≡ engine, on a random sample
        rng = random.Random(20260715)
        sample = rng.sample(check_pool, min(SELF_CHECK_SAMPLES,
                                            len(check_pool)))
        cfg = dict(DEFAULT_CONFIG)
        worst = 0.0
        for key, j, kind, vec, ts in sample:
            node_mask, trace_mask = eng._asof_masks(ts, eng._n)
            pick, enc = eng._episodic_vectors(env.brain, vec, cfg, eng._n,
                                              as_of=ts,
                                              trace_mask=trace_mask)
            mine = episodic_from_sims(eng, env.brain, role_map,
                                      trace_mat @ vec, ts, trace_created)
            for r in range(eng._n):
                m = mine.get(r, (0.0, 0.0))
                worst = max(worst, abs(float(pick[r]) - m[0]),
                            abs(float(enc[r]) - m[1]))
        c['self_check_samples'] = len(sample)
        ok = worst <= TOL
        print('self-check: %d samples, worst |Δ| = %.3g → %s'
              % (len(sample), worst, 'OK' if ok else 'MISMATCH'))
        if not ok:
            walker.execute('DROP TABLE IF EXISTS cand_turn_episodic')
            walker.commit()
            raise SystemExit('episodic_roles: self-check FAILED (worst %.3g '
                             '> %.1g) — the shortcut is NOT the engine; '
                             'table dropped, nothing stamped.' % (worst, TOL))

    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('episodic_roles_version', EPISODIC_VERSION),
         ('episodic_self_check', json.dumps(
             {'samples': len(sample), 'worst_abs_delta': worst}))] +
        [('episodic_' + k, str(v)) for k, v in sorted(c.items())])
    walker.commit()
    walker.close()
    print('counters:')
    for k in sorted(c):
        print('  %-28s %d' % (k, c[k]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
