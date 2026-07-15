"""Walker ↔ engine-as_of cross-check (§20.11 test contract (d)).

Two independent implementations of the same content-lane math — the walker's
per-(turn, candidate, j=0) columns (eval/laf/walker/scores.py, built from raw
SQL over stored vectors) and the production engine's as_of path
(servers/recall_laf.py, caches + masks) — compared ROW-LEVEL over every
labeled turn. Agreement = both almost certainly right. Divergence = one
caught the other's bug: INVESTIGATE, never average.

What is compared, per candidate row at j=0 (holding the walker's stored
q_vec fixed so the lane math itself is what's under test):
    v_<view>_op  ↔  engine mats[view][row] @ q_vec      (6 cosine views)
    sit_op       ↔  engine mats['_situation'][row] @ q_vec
    idf_op       ↔  engine._idf_asof(op_text[:500], n, turn_ts)[row]

The engine side runs the REAL production code (LafV1Engine cache build +
_idf_asof) on an IsolatedBrain copy — never the live DB, never a
reimplementation (leakage-by-reimplementation is the bug class this whole
design kills).

KNOWN divergence classes (counted separately, never folded into the stats).
The walker reads brain.db at build time; the engine reads it NOW — the delta
between those snapshots is live-DB drift, not lane-math disagreement, and it
is provable per node:
  since_build_revised   out-of-tolerance row whose node's updated_at is
                        NEWER than the newest labeled turn (the walker build
                        runs after the last turn by construction, so such a
                        change cannot be in the walker's vectors). Excluded
                        from the verdict, fully listed in the report.
  archived_since_build  candidate with no engine row whose node is now
                        archived=1 (engine is live-only, review F1; this
                        tally is §20.3's since-archived pre-fit measure)
  cand_missing_engine   no engine row and NOT archived — a live node with
                        no embedding views; investigate if nonzero
  walker_null_engine_ok walker cell NULL (vector missing at build) but the
                        engine has one now (backfill/re-embed since)
  walker_ok_engine_nan  walker scored it, engine NaN now (vector gone)
  idf corpus delta      walker df counts ALL live-at-build nodes (titled or
                        not); the engine counts its own title corpus
                        (master ∩ titled), and the archived-since churn
                        shifts rare-token df. Median-gated; the tail is
                        reported, not hidden.

An out-of-tolerance row on a node with NO provable since-build change is a
REAL disagreement — the verdict stays INVESTIGATE no matter how few.

Run:  ./dev python3 eval/laf/walker/cross_check.py
Exit: 0 = AGREE, 1 = INVESTIGATE (report written either way).
"""
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import (open_walker, lanes_version, check_lane_schema,
                       EXTRACT_VERSION, EMBED_VERSION, WALKER_DIR)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import LafV1Engine, MAXSIM_VIEWS, _unit  # noqa: E402

TEXT_CAP = 500                    # scores.py's idf text cap — must match
# cosine lanes: identical stored blobs + identical _unit → only float32
# matmul reassociation separates the two sides. 1e-3 is ~20× above that
# noise and ~20× below the embedder batch-jitter floor (±0.02, d9176bfb).
COSINE_TOL = 1e-3
# idf carries the known corpus-membership delta (see module docstring) —
# gate on the MEDIAN staying tiny and report the tail loudly.
IDF_MEDIAN_TOL = 1e-3
REPORT = WALKER_DIR / 'cross_check.md'


def gate_provenance(walker):
    """Refuse unproven data — the stamps must match this code's expectations
    (same discipline as scores.py / health.py check 0)."""
    stamps = dict(walker.execute(
        "SELECT key, value FROM build_meta WHERE key IN "
        "('extract_version','embed_version','scores_lanes_version')"))
    expect = {'extract_version': EXTRACT_VERSION,
              'embed_version': EMBED_VERSION,
              'scores_lanes_version': lanes_version(MAXSIM_VIEWS)}
    bad = {k: (stamps.get(k), v) for k, v in expect.items()
           if stamps.get(k) != v}
    if bad:
        raise SystemExit('cross_check: stale walker artifact — %s. '
                         'Rebuild (extract → embed → scores), never bypass.'
                         % ', '.join('%s stamped %r expects %r' % (k, a, b)
                                     for k, (a, b) in bad.items()))
    check_lane_schema(walker, MAXSIM_VIEWS)


def build_engine(brain):
    """Production cache build — matrices + titles only (content lanes need
    no trace matrix; episodic lanes are not walker columns)."""
    eng = LafV1Engine()
    with eng._lock:
        eng._refresh_matrices(brain, None)
        eng._refresh_titles(brain)
    return eng


def main():
    walker = open_walker()
    gate_provenance(walker)

    from tests.isolated_brain import IsolatedBrain
    lanes = [('v_%s_op' % v.strip('_'), v) for v in MAXSIM_VIEWS]
    lanes.append(('sit_op', '_situation'))
    lane_cols = [c for c, _ in lanes] + ['idf_op']

    # walker side: labeled turns (q_vec + capped op_text + ts) and their
    # j=0 candidate rows
    turns = {}
    for sess, epoch, seq, ts, opt, qv in walker.execute(
            "SELECT session_id, epoch, seq, ts, op_text, q_vec "
            "FROM turns WHERE labeled=1"):
        turns[(sess, epoch, seq)] = (ts, (opt or '')[:TEXT_CAP],
                                     _unit(qv) if qv else None)
    rows_by_turn = defaultdict(list)
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, %s "
            "FROM cand_turn_scores WHERE j=0" % ', '.join(lane_cols)):
        rows_by_turn[(row[0], row[1], row[2])].append(row[3:])
    walker.close()

    c = defaultdict(int)
    diffs = {name: [] for name, _ in lanes}
    diffs['idf_op'] = []
    worst = defaultdict(lambda: (0.0, None))     # lane → (|Δ|, node_id)
    outliers = defaultdict(set)                  # lane → {node_id} over tol
    missing_nodes = set()
    max_turn_ts = max(t[0] for t in turns.values())

    with IsolatedBrain() as env:
        eng = build_engine(env.brain)
        n = eng._n
        c['engine_rows'] = n
        c['engine_title_corpus'] = len(eng._title_created)
        for key, cand_rows in sorted(rows_by_turn.items()):
            t = turns.get(key)
            if t is None:
                c['scored_turn_not_labeled'] += 1
                continue
            ts, op_text, qv = t
            if qv is None:
                c['turns_no_qvec'] += len(cand_rows)
                continue
            # engine content lanes, THE production code paths
            view_vals = {name: eng._mats[vt][:n] @ qv for name, vt in lanes}
            idf_vec = eng._idf_asof(op_text, n, ts) if op_text else None
            for cand in cand_rows:
                nid, vals = cand[0], cand[1:]
                row = eng._idx.get(nid)
                if row is None:
                    missing_nodes.add(nid)
                    c['cand_missing_engine_rows'] += 1
                    continue
                c['rows_compared'] += 1
                for (name, _), wval in zip(lanes, vals[:len(lanes)]):
                    eval_ = float(view_vals[name][row])
                    if wval is None:
                        if np.isfinite(eval_):
                            c['walker_null_engine_ok'] += 1
                        continue
                    if not np.isfinite(eval_):
                        c['walker_ok_engine_nan'] += 1
                        continue
                    d = abs(float(wval) - eval_)
                    diffs[name].append(d)
                    if d > COSINE_TOL:
                        outliers[name].add(nid)
                    if d > worst[name][0]:
                        worst[name] = (d, nid)
                wval = vals[len(lanes)]
                if wval is not None and idf_vec is not None:
                    d = abs(float(wval) - float(idf_vec[row]))
                    diffs['idf_op'].append(d)
                    if d > worst['idf_op'][0]:
                        worst['idf_op'] = (d, nid)

        # classify drift per node against the SAME snapshot the engine read
        conn = env.brain._nodes.conn
        drift = {}                    # node_id → provable since-build reason
        out_nodes = set().union(*outliers.values()) if outliers else set()
        for nid in out_nodes:
            r = conn.execute('SELECT updated_at FROM nodes WHERE id=?',
                             (nid,)).fetchone()
            if r and (r[0] or '') > max_turn_ts:
                drift[nid] = 'revised %s (> last turn %s)' % (r[0],
                                                              max_turn_ts)
        for nid in missing_nodes:
            r = conn.execute('SELECT archived FROM nodes WHERE id=?',
                             (nid,)).fetchone()
            if r and r[0]:
                c['archived_since_build_nodes'] += 1
            else:
                c['cand_missing_engine_live'] += 1   # investigate if > 0

    # verdicts
    lines = ['# cross_check — engine-as_of ↔ walker content lanes (§20.11 d)',
             '', 'engine commit: %s' % subprocess.run(
                 ['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO,
                 capture_output=True, text=True).stdout.strip(),
             'walker stamps: extract=%s embed=%s lanes=%s' % (
                 EXTRACT_VERSION, EMBED_VERSION, lanes_version(MAXSIM_VIEWS)),
             '']
    verdict_fail = []
    lines.append('| lane | rows | median |Δ| | p99 |Δ| | max |Δ| | worst node | verdict |')
    lines.append('|---|---|---|---|---|---|---|')
    for name in [n_ for n_, _ in lanes] + ['idf_op']:
        arr = np.asarray(diffs[name])
        if not len(arr):
            lines.append('| %s | 0 | — | — | — | — | NO DATA |' % name)
            verdict_fail.append(name + ':no-data')
            continue
        med, p99, mx = (float(np.median(arr)), float(np.percentile(arr, 99)),
                        float(arr.max()))
        if name == 'idf_op':
            ok = med <= IDF_MEDIAN_TOL
            v = 'AGREE' if ok else 'INVESTIGATE (median>tol)'
        else:
            # out-of-tolerance rows are excused ONLY when every one sits on
            # a node with a PROVABLE since-build change; any other node in
            # the outlier set is a real disagreement
            unexplained = outliers[name] - set(drift)
            if mx <= COSINE_TOL:
                ok, v = True, 'AGREE'
            elif not unexplained:
                ok, v = True, 'AGREE (drift-explained: %d nodes)' % len(
                    outliers[name])
            else:
                ok, v = False, ('INVESTIGATE (unexplained: %s)'
                                % ', '.join(sorted(x[:8]
                                                   for x in unexplained)))
        if not ok:
            verdict_fail.append(name)
        lines.append('| %s | %d | %.2e | %.2e | %.2e | %s | %s |'
                     % (name, len(arr), med, p99, mx,
                        worst[name][1] or '—', v))
    lines.append('')
    lines.append('## divergence-class counters (never folded into the stats)')
    for k in sorted(c):
        lines.append('- %s: %d' % (k, c[k]))
    if drift:
        lines.append('')
        lines.append('## since-build drift nodes (excused from the verdict)')
        for nid, why in sorted(drift.items()):
            lines.append('- %s: %s' % (nid[:8], why))
    lines.append('')
    overall = 'AGREE' if not verdict_fail else ('INVESTIGATE: ' +
                                                ', '.join(verdict_fail))
    lines.append('**Overall: %s**' % overall)
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if not verdict_fail else 1


if __name__ == '__main__':
    sys.exit(main())
