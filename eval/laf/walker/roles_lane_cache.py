"""Dense per-turn conn/auth lane cache — the reach-exam substrate.

cand_turn_episodic_roles (episodic_roles_v2.py) is candidate-keyed; the reach
exam ranks the gold among ALL master nodes (field_cache_index.json order), so
the new lanes need the same dense shape lane_cache.npy has. This writes:

  roles_lane_cache.npy   float32 [n_index_turns, 2, n_nodes]
                         lanes: 0=conn, 1=auth; query = op0/q_vec ONLY
                         (F0 current-message scope — the M_h moment stack
                         keeps production lanes; j>=1 role work is the
                         candidate table's job)
  roles_lane_index.json  {'lanes': ['conn','auth'], 'rows': {key: i},
                          'roles_v2_version': ..., 'n_nodes': ...}

Columns follow field_cache_index's master ORDER (mapped by node id — the
build-era master and today's engine master differ by post-build nodes, which
are absent from the exam universe and skipped).

SELF-CHECK: for sampled turns, the dense values at candidate columns must
equal cand_turn_episodic_roles (j=0, *_op) exactly — cross-artifact
consistency with the already-self-checked table. Mismatch → nothing written.

Run:  BRAIN_DB_DIR=~/AgentsContext/brain-snap-roles-20260729 \
      ./dev python3 eval/laf/walker/roles_lane_cache.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker, OUT_DIR
from episodic_roles_v2 import (build_conn_map, build_auth_map,
                               roles_v2_from_sims, select_moments, conn_at,
                               auth_at, ROLES_V2_VERSION)
from episodic_roles import TOL, WINDOW

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit  # noqa: E402

CACHE = OUT_DIR / 'roles_lane_cache.npy'
INDEX = OUT_DIR / 'roles_lane_index.json'
SELF_CHECK_TURNS = 40


def main():
    walker = open_walker()
    stamped = walker.execute("SELECT value FROM build_meta WHERE "
                             "key='episodic_roles_v2_version'").fetchone()
    if not stamped or stamped[0] != ROLES_V2_VERSION:
        raise SystemExit('roles_lane_cache: cand_turn_episodic_roles not '
                         'stamped at %s — run episodic_roles_v2.py first.'
                         % ROLES_V2_VERSION)
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    n_nodes = idx['n_nodes']
    col_of = {nid: i for i, nid in enumerate(idx['master'])}
    turn_rows = idx['turns']

    qv = {}
    ts_of = {}
    for sess, epoch, seq, q, ts in walker.execute(
            'SELECT session_id, epoch, seq, q_vec, ts FROM turns'):
        qv[(sess, epoch, seq)] = _unit(q) if q else None
        ts_of[(sess, epoch, seq)] = ts

    from tests.isolated_brain import IsolatedBrain
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)
        conn_map = build_conn_map(env.brain.conn, env.brain._trace_dal.conn)
        auth_map = build_auth_map(env.brain.conn, env.brain._trace_dal.conn)

        # id-keyed variant of roles_v2_from_sims: dense columns come from the
        # INDEX master, not the engine master (post-build nodes skipped).
        def dense_roles(vec, ts):
            out = np.zeros((2, n_nodes), dtype=np.float32)
            for (sess, _short, stop), s in select_moments(
                    eng, trace_mat @ vec, ts, trace_created).items():
                runs_list = conn_map.get(sess, ())
                stops_map = auth_map.get(sess, {})
                conn_ids, auth_ids = set(), set()
                for ws in range(max(stop - WINDOW, 0), stop + WINDOW + 1):
                    conn_ids.update(conn_at(runs_list, ws, ts))
                    auth_ids.update(auth_at(stops_map, ws, ts))
                for lane, ids in ((0, conn_ids), (1, auth_ids)):
                    for nid in ids:
                        c = col_of.get(nid)
                        if c is not None and s > out[lane, c]:
                            out[lane, c] = s
            return out

        cache = np.zeros((len(turn_rows), 2, n_nodes), dtype=np.float32)
        rows_map, n_noq = {}, 0
        for t in turn_rows:
            key = (t['key'][0], int(t['key'][1]), int(t['key'][2]))
            v, ts = qv.get(key), ts_of.get(key)
            if v is None or ts is None:
                n_noq += 1
                continue
            cache[t['row']] = dense_roles(v, ts)
            rows_map['%s/%d/%d' % key] = t['row']

        # cross-artifact self-check vs the stamped candidate table (j=0, op)
        rng = random.Random(20260729)
        keys = rng.sample(sorted(rows_map), min(SELF_CHECK_TURNS,
                                                len(rows_map)))
        worst = 0.0
        n_cells = 0
        for k in keys:
            sess, epoch, seq = k.split('/')
            for nid, co, ao in walker.execute(
                    'SELECT node_id, conn_op, auth_op FROM '
                    'cand_turn_episodic_roles WHERE session_id=? AND epoch=? '
                    'AND seq=? AND j=0', (sess, int(epoch), int(seq))):
                c = col_of.get(nid)
                if c is None:
                    continue
                r = rows_map[k]
                worst = max(worst, abs(cache[r, 0, c] - (co or 0.0)),
                            abs(cache[r, 1, c] - (ao or 0.0)))
                n_cells += 2
        ok = worst <= TOL
        print('self-check: %d turns / %d cells, worst |Δ| = %.3g → %s'
              % (len(keys), n_cells, worst, 'OK' if ok else 'MISMATCH'))
        if not ok:
            raise SystemExit('roles_lane_cache: cross-artifact MISMATCH — '
                             'nothing written.')

    np.save(CACHE, cache)
    INDEX.write_text(json.dumps({
        'lanes': ['conn', 'auth'], 'n_nodes': n_nodes,
        'roles_v2_version': ROLES_V2_VERSION,
        'master_hash': idx['master_hash'],
        'turns_covered': len(rows_map), 'turns_no_qvec': n_noq}))
    nz = [int((cache[:, i] > 0).sum()) for i in (0, 1)]
    print('DONE %s: %d turns, nonzero cells conn=%d auth=%d'
          % (CACHE.name, len(rows_map), nz[0], nz[1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
