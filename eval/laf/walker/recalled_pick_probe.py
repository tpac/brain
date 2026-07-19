"""Move 1 smoke test — does redefining the episodic PICK lane from SELECTED
(~3-8 Haiku picks per moment) to RECALLED (picked u dropped, the full ~25 that
were available) reach the gold more RELIABLY? (Tom, 2026-07-18: the recalled-25
is denser -> less z-inflation, and it's what the Scribe saw -> encode/decode
alignment.)

Reuses the production episodic machinery (IsolatedBrain + LafV1Engine + the
walker's build_role_map/roles_at) and adds a third `recalled` channel to the
moment loop at j0 (the current message). For each val gold turn we rank the
gold under pick_selected vs pick_recalled, support-z (the sparse-lane norm),
and report reach@5 + avg support. Higher support with equal/better reach =
Tom's reliability win; big support with WORSE reach = flooding.

Run:  ./dev python3 eval/laf/walker/recalled_pick_probe.py
Pool60: BRAIN_DB_DIR=.../0a9baa/pooled WALKER_OUT_DIR=.../0a9baa/walker ./dev python3 ...
"""
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (LafV1Engine, _unit, _zscore_support)  # noqa: E402
from q1_sweep import load, gate_provenance                          # noqa: E402
from episodic_roles import (build_role_map, roles_at, _stop_of,      # noqa: E402
                            TOP_MOMENTS, WINDOW, K_MAX)
from tests.isolated_brain import IsolatedBrain                       # noqa: E402


def episodic_channels(eng, role_map, sims, as_of, trace_created):
    """{node_row: (pick_selected, recalled)} at one query vector — mirrors
    episodic_from_sims' moment scan, but ALSO scores recalled = picked u
    dropped (no picked-wins subtraction)."""
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
        picked, dropped = set(), set()
        for ws in range(max(stop - WINDOW, 0), stop + WINDOW + 1):
            p, _e, d = roles_at(rec, ws, as_of)
            picked |= p
            dropped |= d
        recalled = picked | dropped
        for nid in picked:
            r = eng._resolve(nid)
            if r is not None and s > out.get(r, (0.0, 0.0))[0]:
                out[r] = (s, out.get(r, (0.0, 0.0))[1])
        for nid in recalled:
            r = eng._resolve(nid)
            if r is not None and s > out.get(r, (0.0, 0.0))[1]:
                out[r] = (out.get(r, (0.0, 0.0))[0], s)
    return out


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)
    qvec = {}
    for sess, epoch, seq, qv, ts in walker.execute(
            "SELECT session_id, epoch, seq, q_vec, ts FROM turns "
            "WHERE labeled=1"):
        qvec[(sess, epoch, seq)] = (_unit(qv) if qv else None, ts)
    walker.close()

    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = np.percentile(allsoft, 90)

    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_traces(env.brain)
        trace_created = np.asarray(eng._tr_created, dtype='<U40')
        trace_mat = np.vstack(eng._tr_blocks)
        role_map = build_role_map(env.brain)

        reach_sel = reach_rec = 0
        sup_sel = sup_rec = 0
        both = only_sel = only_rec = neither = 0
        N = 0
        for td in turns:
            if not td.val or not np.isfinite(td.soft).any():
                continue
            g = int(np.nanargmax(td.soft))
            if td.soft[g] < hi or td.key not in qvec:
                continue
            qv, ts = qvec[td.key]
            if qv is None:
                continue
            ch = episodic_channels(eng, role_map, trace_mat @ qv, ts,
                                   trace_created)
            nc = len(td.cands)
            sel = np.zeros(nc)
            rec = np.zeros(nc)
            for i, nid in enumerate(td.cands):
                r = eng._resolve(nid)
                if r is not None and r in ch:
                    sel[i], rec[i] = ch[r]
            N += 1
            sup_sel += int((sel != 0).sum())
            sup_rec += int((rec != 0).sum())
            zs, zr = _zscore_support(sel, nc), _zscore_support(rec, nc)
            rs = int((zs > zs[g]).sum()) < 5
            rr = int((zr > zr[g]).sum()) < 5
            reach_sel += int(rs)
            reach_rec += int(rr)
            both += int(rs and rr)
            only_sel += int(rs and not rr)
            only_rec += int(rr and not rs)
            neither += int(not rs and not rr)

        print('val gold turns: %d   (soft hi=%.2f)' % (N, hi))
        print('\nlane            reach@5      avg support/pool')
        print('  pick_selected  %4.0f%% (%d)   %.1f'
              % (100 * reach_sel / N, reach_sel, sup_sel / N))
        print('  pick_recalled  %4.0f%% (%d)   %.1f'
              % (100 * reach_rec / N, reach_rec, sup_rec / N))
        print('\noverlap: both %d · only_selected %d · only_recalled %d · '
              'neither %d' % (both, only_sel, only_rec, neither))
    return 0


if __name__ == '__main__':
    sys.exit(main())
