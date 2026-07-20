"""Phase 1 of the mesh work (Tom go, 2026-07-20): cache per-msg SETTLED
FIELDS (full-graph, composed LAF per slot) for every gold turn, so the
field-level oracle, the readout menu, and every mesh-formula sweep run as
free re-fits over the cache — the expensive-substrate-once design.

A slot's field = Σ_lane gain·z(lane_j, node_mask) over ALIVE nodes at the
turn's ts — the within-LAF mesh at production GAINS, one composed field per
(turn, slot). Slots: op0 (q_vec), op1, anchor1, op2 (J_LIMIT=2 parity with
moment_influence.turn_fields; j0-anchor never a cue — temporal-leak rule).

Z ROUTING (deliberate, NOT the shipped default): sparse zero-sea lanes
(pick/enc/idf) go through support-z, dense lanes through current-z — the
engine's own per-lane gating under the P3.0 'support' variant. Production
ships z_norm='current' (plain z everywhere), but at full-field scale the
zero seas explode under plain z (enc z≈11 vs cosine z≈2, node 0ccc5481) —
fitting meshes on an exploded substrate is worse than the variant gap.
Stamped in the index meta as 'sparse_z' so downstream knows what it reads.

Output (WALKER_DIR): field_cache.npy [n_turns × 4 slots × n_nodes] float32,
NaN = masked/missing; field_cache_index.json (turn keys + row pointers,
gold/cand/sel rows, slot coverage, engine master hash).

Self-checks (printed; hard-fail on parity):
  1. raw parity — candidate-restricted op0 maxsim/sit vs cand_turn_scores
  1b. COMPOSED base-parity (reach_leg discipline) — K0 composition under
      production z ≡ eng.scores(as_of) top-25 sequence, hard-fail
  2. time-mask — nodes created after ts are NaN (negative test per turn)
  3. coverage — % turns with each slot present

Run:   ./dev python3 eval/laf/walker/field_cache_build.py [--smoke N]
Pool60: BRAIN_DB_DIR=... WALKER_OUT_DIR=... (same as every walker build)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import zscore_variant, _unit                # noqa: E402
from q1_sweep import (GAINS, load, gate_provenance, configs,        # noqa: E402
                      weights, compose, stack_messages)
from moment_influence import turn_fields                            # noqa: E402
from reach_leg import rank_rows, TEXT_CAP                           # noqa: E402
from tests.isolated_brain import IsolatedBrain                       # noqa: E402

SLOTS = (('op', 0), ('op', 1), ('anchor', 1), ('op', 2))
SPARSE_Z = 'support'              # zero-sea lanes; see Z ROUTING above
CACHE = WALKER_DIR / 'field_cache.npy'
INDEX = WALKER_DIR / 'field_cache_index.json'
PARITY_TURNS = 20
PARITY_TOL = 1e-6                 # same engine, same vectors — exact


def compose_slot(op, an, j, kind, n, mask):
    """One slot's composed field: gain-weighted z over alive nodes."""
    src = op if kind == 'op' else an
    if np.isnan(src['maxsim'][:, j]).all():
        return np.full(n, np.nan, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    for ln, g in GAINS.items():
        col = src[ln][:, j]
        if np.isnan(col).all():
            continue
        sparse = ln in ('pick', 'enc', 'idf')
        out += np.float32(g) * zscore_variant(
            np.where(np.isfinite(col), col, 0.0) if sparse else col,
            n, mask=mask, kind=SPARSE_Z if sparse else 'current')
    out[~mask] = np.nan
    return out


def main():
    smoke = 0
    if '--smoke' in sys.argv:
        smoke = int(sys.argv[sys.argv.index('--smoke') + 1])
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    tmeta = dict(((s, e, q), (stop, ts, txt, qv)) for s, e, q, stop, ts, txt,
                 qv in walker.execute(
                     "SELECT session_id, epoch, seq, stop, ts, op_text, "
                     "q_vec FROM turns WHERE labeled=1"))
    # stored op0 lanes for the parity check
    stored = {}
    for s, e, q, nid, v1, v2, v3, v4, v5, v6, sit in walker.execute(
            "SELECT session_id, epoch, seq, node_id, v_title_op, "
            "v_primary_op, v_high_meta_op, v_other_meta_op, "
            "v_edge_context_op, v_question_op, sit_op "
            "FROM cand_turn_scores WHERE j=0"):
        stored[(s, e, q, nid)] = (
            np.nanmax([x for x in (v1, v2, v3, v4, v5, v6)
                       if x is not None] or [np.nan]), sit)
    walker.close()

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = float(np.percentile(allsoft, 90))
    gold_turns = []
    for td in turns:
        if not np.isfinite(td.soft).any():
            continue
        g = int(np.nanargmax(td.soft))
        if td.soft[g] >= hi and td.key in tmeta \
                and _unit(tmeta[td.key][3]) is not None:
            gold_turns.append((td, g))
    if smoke:
        gold_turns = gold_turns[:smoke]
    print('gold turns to cache: %d (hi=%.2f)' % (len(gold_turns), hi))

    from servers.recall_laf import LafV1Engine
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        eng._brain_ref = env.brain
        n = eng._n
        master_hash = hashlib.sha256(
            ('|'.join(eng._master[:n])).encode()).hexdigest()[:16]

        fields = np.lib.format.open_memmap(
            str(CACHE) + ('.smoke' if smoke else ''), mode='w+',
            dtype=np.float32, shape=(len(gold_turns), len(SLOTS), n))
        fields[:] = np.nan
        index, cov = [], {('%s%d' % (k, j)): 0 for k, j in SLOTS}
        worst_par, par_checked, base_par_ok = 0.0, 0, 0
        cfg_k0, w0 = configs()[0], weights(configs()[0])
        import time
        t0 = time.time()
        for ti, (td, g) in enumerate(gold_turns):
            stop, ts, op_text, qb = tmeta[td.key]
            q0 = _unit(qb)
            if q0 is None:
                # row ti stays all-NaN; index MUST still carry an aligned
                # entry or every later row misaligns with its index record
                index.append({'key': list(td.key), 'row': ti,
                              'skipped': True})
                continue
            node_mask, trace_mask = eng._asof_masks(ts, n)
            op, an = turn_fields(eng, trace_mask, td.key[0], stop, ts,
                                 op_text, q0)
            # -- self-check 1: parity of raw op0 lanes vs stored substrate
            if par_checked < PARITY_TURNS:
                for i, nid in enumerate(td.cands):
                    r = eng._resolve(nid)
                    kk = (*td.key, nid)
                    if r is None or kk not in stored:
                        continue
                    sm, ss = stored[kk]
                    if np.isfinite(sm) and np.isfinite(op['maxsim'][r, 0]):
                        worst_par = max(worst_par,
                                        abs(op['maxsim'][r, 0] - sm))
                    if ss is not None and np.isfinite(op['sit'][r, 0]):
                        worst_par = max(worst_par, abs(op['sit'][r, 0] - ss))
                # -- self-check 1b: composed base-parity (reach_leg
                #    discipline) — K0 composition under PRODUCTION z ≡
                #    eng.scores(as_of) top-25 sequence
                mats, ww = {}, None
                for ln in GAINS:
                    mats[ln], ww = stack_messages(op[ln], an[ln], w0,
                                                  cfg_k0)
                s_k0 = compose(mats, ww, cfg_k0, n, mask=node_mask)
                order = rank_rows(s_k0, node_mask)
                smap, _tel = eng.scores(env.brain,
                                        (op_text or '')[:TEXT_CAP], q0,
                                        as_of=ts)
                eng_top = [nid for nid, _s in sorted(
                    smap.items(), key=lambda kv: -kv[1])][:25]
                mine = [eng._master[r] for r in order[:25]]
                assert mine == eng_top, \
                    'BASE-PARITY MISMATCH at %s' % (td.key,)
                base_par_ok += 1
                par_checked += 1
            # -- compose + store per slot
            for si, (kind, j) in enumerate(SLOTS):
                f = compose_slot(op, an, j, kind, n, node_mask)
                if not np.isnan(f).all():
                    cov['%s%d' % (kind, j)] += 1
                fields[ti, si, :] = f
            # -- self-check 2: time-mask negative test
            dead = np.flatnonzero(~node_mask)
            if len(dead):
                assert np.isnan(fields[ti, 0, dead[0]]), \
                    'time-mask leak at turn %s' % (td.key,)
            cand_rows = [eng._resolve(nid) for nid in td.cands]
            index.append({
                'key': list(td.key), 'row': ti, 'ts': ts, 'gold_i': g,
                'cand_rows': [(-1 if r is None else int(r))
                              for r in cand_rows],
                'sel': td.sel.astype(int).tolist(),
                'soft': [None if not np.isfinite(x) else round(float(x), 4)
                         for x in td.soft],
                'alive': int(node_mask.sum()),
            })
            if (ti + 1) % 100 == 0:
                el = time.time() - t0
                print('  %d/%d  (%.2fs/turn, ETA %.0fm)'
                      % (ti + 1, len(gold_turns), el / (ti + 1),
                         el / (ti + 1) * (len(gold_turns) - ti - 1) / 60))
        fields.flush()

    assert worst_par < PARITY_TOL, \
        'PARITY FAIL: worst |Δ| %.4f >= %s' % (worst_par, PARITY_TOL)
    print('raw parity (%d turns): worst |Δ| %.6f  OK · composed '
          'base-parity: %d/%d exact' % (par_checked, worst_par,
                                        base_par_ok, par_checked))
    print('slot coverage:', {k: '%d (%.0f%%)' % (v, 100 * v /
                                                 max(1, len(gold_turns)))
                             for k, v in cov.items()})
    (Path(str(INDEX) + ('.smoke' if smoke else ''))).write_text(json.dumps({
        'slots': ['%s%d' % (k, j) for k, j in SLOTS], 'n_nodes': n,
        'master_hash': master_hash, 'soft_hi': hi, 'sparse_z': SPARSE_Z,
        'dtype': 'float32', 'turns': index}))
    print('wrote %s (%d turns × %d slots × %d nodes)'
          % (CACHE, len(gold_turns), len(SLOTS), n))
    return 0


if __name__ == '__main__':
    sys.exit(main())
