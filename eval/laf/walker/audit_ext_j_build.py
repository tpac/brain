"""Foundations audit (c): the ext episodic channels were built at the j0 cue
(q_vec) only — build the SAME channels cued at history slots (op j1, anchor
j1, op j2; slot semantics identical to scores.py: seq-j within the epoch,
document-side vectors) so the composition gate can test whether episodic
signal enters better through history cues.

Non-destructive: writes cand_turn_episodic_ext_j (slot column), reuses
channels() from episodic_ext_build verbatim; as_of stays the CURRENT turn's
ts (we stand at the current moment, cuing with older messages).

Run: ./dev python3 eval/laf/walker/audit_ext_j_build.py   (pool60 via env)
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, _unit                   # noqa: E402
from episodic_roles import build_role_map                           # noqa: E402
from episodic_ext_build import channels                             # noqa: E402
from tests.isolated_brain import IsolatedBrain                       # noqa: E402

SLOTS = (('op', 1), ('anchor', 1), ('op', 2))


def main():
    walker = open_walker()
    epoch_turns = {}
    for s, e, q, ov, av, qv, ts, lab in walker.execute(
            "SELECT session_id, epoch, seq, op_vec, anchor_vec, q_vec, ts, "
            "labeled FROM turns"):
        epoch_turns[(s, e, q)] = (_unit(ov) if ov else None,
                                  _unit(av) if av else None, ts, lab)
    cand_by_turn = defaultdict(list)
    for s, e, q, nid in walker.execute(
            "SELECT c.session_id, c.epoch, c.seq, c.node_id FROM candidates c "
            "JOIN turns t ON t.session_id=c.session_id AND t.epoch=c.epoch "
            " AND t.seq=c.seq WHERE t.labeled=1 AND c.node_id IS NOT NULL"):
        cand_by_turn[(s, e, q)].append(nid)

    walker.execute('DROP TABLE IF EXISTS cand_turn_episodic_ext_j')
    walker.execute(
        'CREATE TABLE cand_turn_episodic_ext_j ('
        ' session_id TEXT, epoch INTEGER, seq INTEGER, node_id TEXT,'
        ' slot TEXT, pick_rec REAL, enc_graph REAL,'
        ' PRIMARY KEY (session_id, epoch, seq, node_id, slot))')

    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)
        role_map = build_role_map(env.brain)
        adj = defaultdict(set)
        for s, t in env.brain.conn.execute(
                "SELECT e.source_id, e.target_id FROM edges e "
                "JOIN nodes ns ON ns.id=e.source_id "
                "JOIN nodes nt ON nt.id=e.target_id "
                "WHERE ns.archived=0 AND nt.archived=0"):
            adj[s].add(t)
            adj[t].add(s)

        buf, done = [], 0
        for key in sorted(cand_by_turn):
            sess, epoch, seq = key
            ts = epoch_turns[key][2]
            for side, j in SLOTS:
                src = epoch_turns.get((sess, epoch, seq - j))
                if src is None:
                    continue
                cue = src[0] if side == 'op' else src[1]
                if cue is None:
                    continue
                prec, egraph = channels(eng, role_map, trace_mat @ cue, ts,
                                        trace_created, adj, 1, 1.0)
                slot = '%s%d' % (side, j)
                for nid in cand_by_turn[key]:
                    r = eng._resolve(nid)
                    pr = prec.get(r, 0.0) if r is not None else 0.0
                    eg = egraph.get(r, 0.0) if r is not None else 0.0
                    if pr or eg:
                        buf.append((sess, epoch, seq, nid, slot, pr, eg))
            done += 1
            if len(buf) >= 20000:
                walker.executemany(
                    'INSERT OR REPLACE INTO cand_turn_episodic_ext_j VALUES '
                    '(?,?,?,?,?,?,?)', buf)
                walker.commit()
                buf = []
        if buf:
            walker.executemany(
                'INSERT OR REPLACE INTO cand_turn_episodic_ext_j VALUES '
                '(?,?,?,?,?,?,?)', buf)
        walker.execute("INSERT OR REPLACE INTO build_meta VALUES "
                       "('episodic_ext_j_version', 'ext-j-v1-hop1-decay1')")
        walker.commit()
        tot = walker.execute("SELECT COUNT(*) FROM "
                             "cand_turn_episodic_ext_j").fetchone()[0]
        by_slot = walker.execute(
            "SELECT slot, COUNT(*), SUM(pick_rec>0), SUM(enc_graph>0) "
            "FROM cand_turn_episodic_ext_j GROUP BY slot").fetchall()
        print('built cand_turn_episodic_ext_j: %d turns, %d nonzero rows'
              % (done, tot))
        for slot, n, npr, neg in by_slot:
            print('  %s: %d rows · pick_rec>0 %d · enc_graph>0 %d'
                  % (slot, n, npr, neg))
    walker.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
