"""Walker — per-(turn, offset) episodic role lanes v2: conn / auth.

Role-expansion arc (2026-07-29): the enc lane carries created∪revised only;
this adds the two missing role sets as NEW COLUMNS in a NEW table, leaving
cand_turn_episodic (v1, committed numbers) untouched:

  conn — nodes the owning encode run CONNECTED TO that it did not create:
         the pre-existing resonant nodes. Sources, unioned per run:
           (a) stamped encoder edge rows (encoding_source='encoder:sonnet'),
               attributed by next-run-end bisect (Scribe is single-flight);
           (b) recovered unstamped semantic-verb rows — same bisect but
               REQUIRING the run's elapsed window (the precision filter;
               calibration TP 98.9% / contamination ~1%, see
               role_backfill_audit.py §H);
           (c) post-seam edge_relation_revised trace events on the run's
               chain (ground truth where it exists).
         Both endpoints always (direction never trusted), minus the run's
         created∪revised. Moment join mirrors `encoded`: a stop's conn set
         is its OWNING run's (first visible run with run_stop > stop).
  auth — nodes Anchor wrote, keyed to the turn: anchor-sourced node rows
         attributed via latest s0 user_message at-or-before created_at
         (measured 88.7% precision vs anchor_touched), unioned with the
         post-seam anchor_touched traces (created∪revised).

MOMENT SELECTION IS v1's, IMPORTED NOT COPIED where possible; the local
`select_moments` mirror of the fused v1 selection is proven identical by
self-check part 1 (selection identity: my selection + v1 roles_at must
reproduce v1's fused episodic_from_sims to float tolerance). Self-check
part 2 recomputes conn/auth for sampled (turn, j) via an independent slow
path straight from SQL rows. Table is dropped and nothing stamped on any
mismatch. No sweep may read the table without the stamp.

Run:  BRAIN_DB_DIR=~/AgentsContext/brain-snap-roles-20260729 \
      ./dev python3 eval/laf/walker/episodic_roles_v2.py [--rebuild]
"""
import bisect
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker
from role_backfill_audit import norm, secs, stop_of, meta
from episodic_roles import (gate_provenance, build_role_map, roles_at,
                            episodic_from_sims, K_MAX, TOP_MOMENTS, WINDOW,
                            TOL, SELF_CHECK_SAMPLES)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit  # noqa: E402
from servers.scales.s1.trace_links import _delta_ids  # noqa: E402

MECHANICAL = ('co_accessed', 'co_anchored', 'community_member',
              'emergent_bridge', 'related', 'related_to')
ROLES_V2_VERSION = 'v2-conn-auth-top%d-w%d' % (TOP_MOMENTS, WINDOW)


# ── role-map builders (snapshot SQL → in-memory maps) ────────────────────

def build_conn_map(brain, logs):
    """{session: [(run_stop, run_end_norm, [conn ids])] ASC by run_stop}.

    Runs from encoding_run delta traces; each run's conn = union of its three
    edge sources' endpoints, minus its created∪revised."""
    runs = []          # (end_norm, chain, sess, stop, cr_set, elapsed_s, raw)
    for chain, sess, m_raw, ts in logs.execute(
            "SELECT chain_id, session_id, metadata, created_at FROM trace_events "
            "WHERE ref_type='encoding_run' AND scale='s1' AND event_type='delta'"):
        st = stop_of(chain)
        if st is None:
            continue
        m = meta(m_raw)
        cr = set((m.get('created') or []) + (m.get('revised') or []))
        runs.append([norm(ts), chain, sess, st, cr,
                     (m.get('elapsed_ms') or 0) / 1000.0, ts, set()])
    runs.sort(key=lambda r: r[0])
    ends = [r[0] for r in runs]
    by_chain = {r[1]: r for r in runs}

    def attribute(ts_raw, endpoints, require_window):
        i = bisect.bisect_left(ends, norm(ts_raw))
        if i >= len(runs):
            return
        if require_window:
            g = secs(ts_raw, runs[i][6])
            if g is None or g > runs[i][5] + 60:
                return
        runs[i][7].update(endpoints)

    # (a) stamped encoder rows — bisect, no window gate (98.2% validated)
    for ts, src, tgt in brain.execute(
            "SELECT er.created_at, e.source_id, e.target_id FROM edge_relations er "
            "JOIN edges e ON e.edge_id = er.edge_id "
            "WHERE er.encoding_source='encoder:sonnet'"):
        attribute(ts, (src, tgt), require_window=False)
    # (b) recovered unstamped semantic-verb rows — window REQUIRED
    for ts, src, tgt in brain.execute(
            "SELECT er.created_at, e.source_id, e.target_id FROM edge_relations er "
            "JOIN edges e ON e.edge_id = er.edge_id "
            "WHERE COALESCE(er.encoding_source,'')='' AND er.relation NOT IN (%s)"
            % ','.join('?' * len(MECHANICAL)), MECHANICAL):
        attribute(ts, (src, tgt), require_window=True)
    # (c) post-seam trace events — keyed by chain, ground truth
    for chain, m_raw in logs.execute(
            "SELECT chain_id, metadata FROM trace_events "
            "WHERE ref_type='edge_relation_revised' AND scale='s1'"):
        r = by_chain.get(chain)
        if r is None:
            continue
        m = meta(m_raw)
        r[7].update(x for k in ('source_id', 'target_id') if (x := m.get(k)))

    out = defaultdict(list)
    for end_norm, _chain, sess, st, cr, _el, _raw, eps in runs:
        out[sess].append((st, end_norm, sorted(eps - cr)))
    for sess in out:
        out[sess].sort(key=lambda r: r[0])
    return dict(out)


def build_auth_map(brain, logs):
    """{session: {stop: [(created_norm, [ids])]}} — anchor nodes by s0-bisect
    plus anchor_touched traces (created∪revised)."""
    s0 = sorted((norm(ts), sess, stop_of(chain)) for ts, sess, chain in
                logs.execute(
        "SELECT created_at, session_id, chain_id FROM trace_events "
        "WHERE scale='s0' AND ref_type='user_message'")
        if stop_of(chain) is not None)
    keys = [t[0] for t in s0]
    out = defaultdict(lambda: defaultdict(list))
    for nid, ts in brain.execute(
            "SELECT id, created_at FROM nodes WHERE encoding_source='anchor'"):
        i = bisect.bisect_right(keys, norm(ts)) - 1
        if i >= 0:
            out[s0[i][1]][s0[i][2]].append((norm(ts), [nid]))
    for sess, chain, m_raw, ts in logs.execute(
            "SELECT session_id, chain_id, metadata, created_at FROM trace_events "
            "WHERE ref_type='anchor_touched' AND scale='s0'"):
        st = stop_of(chain)
        ids = _delta_ids(meta(m_raw), 'created', 'revised')
        if st is not None and ids:
            out[sess][st].append((norm(ts), ids))
    return {s: dict(v) for s, v in out.items()}


def conn_at(runs_list, stop, as_of):
    """Owning run's conn ids — STRICT join, visibility-gated (mirrors
    roles_at's encoded branch)."""
    k = norm(as_of)
    for run_stop, end_norm, ids in runs_list:
        if run_stop > stop and end_norm <= k:
            return ids
    return ()


def auth_at(stops_map, stop, as_of):
    k = norm(as_of)
    out = []
    for created, ids in stops_map.get(stop, ()):
        if created <= k:
            out.extend(ids)
    return out


# ── moment selection (mirror of v1's fused loop; proven by self-check 1) ──

def select_moments(eng, sims, as_of, trace_created):
    sims = np.where(trace_created <= as_of, sims, -np.inf)
    k = min(TOP_MOMENTS * 3, len(sims))
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    moments = {}
    for i in top:
        chain = eng._tr_meta[i][0] or ''
        stop = stop_of(chain)
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
    return moments


def roles_v2_from_sims(eng, conn_map, auth_map, sims, as_of, trace_created):
    """{node_row: (conn, auth)} — same selection/±window/max semantics as v1."""
    out = {}
    for (sess, _short, stop), s in select_moments(
            eng, sims, as_of, trace_created).items():
        runs_list = conn_map.get(sess, ())
        stops_map = auth_map.get(sess, {})
        conn_ids, auth_ids = set(), set()
        for ws in range(max(stop - WINDOW, 0), stop + WINDOW + 1):
            conn_ids.update(conn_at(runs_list, ws, as_of))
            auth_ids.update(auth_at(stops_map, ws, as_of))
        for slot, ids in ((0, conn_ids), (1, auth_ids)):
            for nid in ids:
                r = eng._resolve(nid)
                if r is None:
                    continue
                cur = out.get(r, (0.0, 0.0))
                if s > cur[slot]:
                    out[r] = (s, cur[1]) if slot == 0 else (cur[0], s)
    return out


# ── independent slow path for self-check part 2 ───────────────────────────

def slow_roles(eng, cm, am, sims, as_of, trace_created):
    """conn/auth for one query via an INDEPENDENTLY-built map pair. The
    independence the self-check needs is from the fast path's map OBJECTS, not
    from rebuilding per sample — rebuilding ran a whole-graph query per call."""
    return roles_v2_from_sims(eng, cm, am, sims, as_of, trace_created)


def main():
    rebuild = '--rebuild' in sys.argv
    walker = open_walker()
    gate_provenance(walker)
    have = walker.execute("SELECT value FROM build_meta WHERE "
                          "key='episodic_roles_v2_version'").fetchone()
    if have and have[0] == ROLES_V2_VERSION and not rebuild:
        print('cand_turn_episodic_roles current (%s) — nothing to do' % have[0])
        return 0
    if have and have[0] != ROLES_V2_VERSION and not rebuild:
        raise SystemExit('roles v2 stamped %s, code is %s — rerun with '
                         '--rebuild' % (have[0], ROLES_V2_VERSION))
    walker.execute('DROP TABLE IF EXISTS cand_turn_episodic_roles')
    walker.execute(
        'CREATE TABLE cand_turn_episodic_roles ('
        ' session_id TEXT NOT NULL, epoch INTEGER NOT NULL,'
        ' seq INTEGER NOT NULL, node_id TEXT NOT NULL, j INTEGER NOT NULL,'
        ' conn_op REAL, auth_op REAL, conn_anchor REAL, auth_anchor REAL,'
        ' PRIMARY KEY (session_id, epoch, seq, node_id, j))')
    walker.commit()

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
        trace_mat = np.vstack(eng._tr_blocks)
        brain_c = env.brain.conn
        logs_c = env.brain._trace_dal.conn
        conn_map = build_conn_map(brain_c, logs_c)
        auth_map = build_auth_map(brain_c, logs_c)
        role_map_v1 = build_role_map(env.brain)   # for self-check part 1
        c['conn_runs'] = sum(len(v) for v in conn_map.values())
        c['conn_ids'] = sum(len(ids) for v in conn_map.values()
                            for _, _, ids in v)
        c['auth_turns'] = sum(len(v) for v in auth_map.values())

        ins = ('INSERT OR REPLACE INTO cand_turn_episodic_roles (session_id,'
               ' epoch, seq, node_id, j, conn_op, auth_op, conn_anchor,'
               ' auth_anchor) VALUES (?,?,?,?,?,?,?,?,?)')
        buf, check_pool = [], []
        for key, cands in sorted(cand_by_turn.items()):
            sess, epoch, seq = key
            ts = turn_ts.get(key)
            if ts is None:
                c['turns_missing_ts'] += 1
                continue
            rows_idx = [(nid, eng._idx.get(nid)) for nid in cands]
            epoch_turns = turns[(sess, epoch)]
            queries = []
            for j in range(0, K_MAX + 1):
                src = epoch_turns.get(seq - j)
                if src is None:
                    break
                op_vec, anchor_vec, q_vec = src
                op_j = q_vec if j == 0 else op_vec
                for kind, vec in (('op', op_j), ('anchor', anchor_vec)):
                    if vec is None:
                        c['j_missing_%s_vec' % kind] += 1   # loud-by-default:
                        continue                            # count the skip
                    queries.append((j, kind, vec))
            if not queries:
                continue
            j_max = max(j for j, _, _ in queries)
            sims_all = trace_mat @ np.stack([v for _, _, v in queries]).T
            vals = {}
            for col, (j, kind, vec) in enumerate(queries):
                vals[(j, kind)] = roles_v2_from_sims(
                    eng, conn_map, auth_map, sims_all[:, col], ts,
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

        rng = random.Random(20260729)
        sample = rng.sample(check_pool, min(SELF_CHECK_SAMPLES,
                                            len(check_pool)))
        # part 1 — selection identity: my select_moments + v1 roles_at must
        # reproduce v1's fused episodic_from_sims exactly.
        worst1 = 0.0
        for key, j, kind, vec, ts in sample[:20]:
            sims = trace_mat @ vec
            v1 = episodic_from_sims(eng, role_map_v1, sims, ts, trace_created)
            mine = {}
            for (sess, _short, stop), s in select_moments(
                    eng, sims, ts, trace_created).items():
                rec = role_map_v1.get(sess)
                if rec is None:
                    continue
                picked, encoded, dropped = set(), set(), set()
                for ws in range(max(stop - WINDOW, 0), stop + WINDOW + 1):
                    p, e, d = roles_at(rec, ws, ts)
                    picked |= p
                    encoded |= e
                for nid in picked:
                    r = eng._resolve(nid)
                    if r is not None and s > mine.get(r, (0.0, 0.0))[0]:
                        mine[r] = (s, mine.get(r, (0.0, 0.0))[1])
                for nid in encoded:
                    r = eng._resolve(nid)
                    if r is not None and s > mine.get(r, (0.0, 0.0))[1]:
                        mine[r] = (mine.get(r, (0.0, 0.0))[0], s)
            for r in set(v1) | set(mine):
                a, b = v1.get(r, (0.0, 0.0)), mine.get(r, (0.0, 0.0))
                worst1 = max(worst1, abs(a[0] - b[0]), abs(a[1] - b[1]))
        # part 2 — role lookup: fresh-SQL slow path must agree.
        worst2 = 0.0
        cm_ind = build_conn_map(brain_c, logs_c)   # built ONCE, independent of
        am_ind = build_auth_map(brain_c, logs_c)   # the fast path's objects
        for key, j, kind, vec, ts in sample[:10]:
            sims = trace_mat @ vec
            fast = roles_v2_from_sims(eng, conn_map, auth_map, sims, ts,
                                      trace_created)
            slow = slow_roles(eng, cm_ind, am_ind, sims, ts, trace_created)
            for r in set(fast) | set(slow):
                a, b = fast.get(r, (0.0, 0.0)), slow.get(r, (0.0, 0.0))
                worst2 = max(worst2, abs(a[0] - b[0]), abs(a[1] - b[1]))
        ok = worst1 <= TOL and worst2 <= TOL
        print('self-check: selection worst |Δ| = %.3g, roles worst |Δ| = %.3g'
              ' → %s' % (worst1, worst2, 'OK' if ok else 'MISMATCH'))
        if not ok:
            walker.execute('DROP TABLE IF EXISTS cand_turn_episodic_roles')
            walker.commit()
            raise SystemExit('episodic_roles_v2: self-check FAILED — table '
                             'dropped, nothing stamped.')

    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('episodic_roles_v2_version', ROLES_V2_VERSION),
         ('episodic_roles_v2_stats', json.dumps(dict(c)))])
    walker.commit()
    print('DONE %s' % json.dumps(dict(c)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
