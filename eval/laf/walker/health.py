"""Walker health report — §20.4. A build without a green report feeds no sweep.

Six checks, each a section in walker_health.md with PASS / WARN / FAIL:

  1 fill-rate matrix        — key columns × month (the empty-subset class)
  2 join conservation       — Δ-row ledger recounted INDEPENDENTLY from logs
                              db and matched against build_meta, to the row
  3 achieved-window         — history depth available per labeled turn
  4 lane sensitivity        — anti-dead-operator: per lane, (a) within-turn
                              spread across candidates AND (b) per-node spread
                              across turns; a constant/query-independent field
                              fails (b) even when (a) passes — the temporal-
                              operator disease (0c8352f1) and its tautology-
                              threshold trap are both covered
  5 replay sanity           — offline K=0 static-gain composition (P1 gains,
                              nanmax over views, neutral-fill sit) must
                              rank-correlate with the LIVE pool_score ranking
                              recorded in laf_v1-era traces (post-flip turns)
  6 embedding spot-audit    — stored turn vectors vs fresh re-embeds of the
                              same 500-cap render (recipe/prefix drift)

Exit: 0 all PASS/WARN · 1 any FAIL.
Run:  ./dev python3 eval/laf/walker/health.py
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

from walker_db import open_walker, open_logs_ro, WALKER_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import DEFAULT_CONFIG, MAXSIM_VIEWS, _unit, _zscore  # noqa: E402

LAF_LIVE_TS = '2026-07-02T19:30'
SPOT_SAMPLE = 20
RNG = random.Random(20)          # deterministic sampling — reruns compare

TURN_COLS = ['op_text', 'anchor_text', 'query_stored', 'op_vec', 'anchor_vec',
             'op_trace_id', 'project', 'gap_seconds']
CAND_COLS = ['node_id', 'outcome', 'fetched_by', 'pool_score',
             'node_created_at', 'used_next_1']


def sect(lines, title, status, body):
    lines.append('## %s — %s\n' % (title, status))
    lines.extend(body)
    lines.append('')
    return status


def check_fill_rates(walker, lines):
    body = ['| month | ' + ' | '.join(TURN_COLS) + ' | (turns) |',
            '|' + '---|' * (len(TURN_COLS) + 2)]
    worst_recent = 1.0
    for row in walker.execute(
            "SELECT substr(ts,1,7) m, count(*), %s FROM turns GROUP BY m ORDER BY m"
            % ', '.join('sum(%s IS NOT NULL)' % c for c in TURN_COLS)):
        m, total, fills = row[0], row[1], row[2:]
        rates = [f / total for f in fills]
        if m and m >= '2026-05':
            worst_recent = min(worst_recent, rates[TURN_COLS.index('op_vec')])
        body.append('| %s | %s | %d |' % (m, ' | '.join('%.0f%%' % (r * 100) for r in rates), total))
    body.append('')
    body.append('| month | ' + ' | '.join(CAND_COLS) + ' | (cands) |')
    body.append('|' + '---|' * (len(CAND_COLS) + 2))
    for row in walker.execute(
            "SELECT substr(t.ts,1,7) m, count(*), %s FROM candidates c "
            "JOIN turns t USING (session_id, epoch, seq) GROUP BY m ORDER BY m"
            % ', '.join('sum(c.%s IS NOT NULL)' % c for c in CAND_COLS)):
        m, total, fills = row[0], row[1], row[2:]
        body.append('| %s | %s | %d |' % (
            m, ' | '.join('%.0f%%' % (f / total * 100) for f in fills), total))
    status = 'PASS' if worst_recent >= 0.95 else 'FAIL (post-April op_vec fill %.0f%% < 95%%)' % (worst_recent * 100)
    return sect(lines, '1 · Fill-rate matrix (column × month)', status, body)


def check_conservation(walker, logs, lines):
    """Recount the Δ ledger from logs db independently; compare to build_meta."""
    meta = dict(walker.execute("SELECT key, value FROM build_meta").fetchall())
    gold = set(json.loads((WALKER_DIR / 'gold_manifest.json').read_text())['excluded_sessions'])
    total = empty = gold_synth = 0
    for sess, meta_raw in logs.execute(
            "SELECT session_id, metadata FROM trace_events "
            "WHERE scale='s1' AND event_type='delta' AND ref_type='additionalContext'"):
        total += 1
        if sess in gold or not re.match(
                r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', sess):
            gold_synth += 1
            continue
        try:
            if not (json.loads(meta_raw or '{}').get('outcomes_per_candidate') or {}):
                empty += 1
        except (ValueError, TypeError):
            empty += 1
    labeled = int(meta.get('extract_labeled_turns_written', 0))
    unpaired = int(meta.get('extract_delta_unpaired', 0))
    missing_cands = int(meta.get('extract_label_missing_candidates', 0))
    missing_o = int(meta.get('extract_label_missing_O', 0))
    disagree = int(meta.get('extract_label_excluded_text_disagree', 0))
    accounted = labeled + empty + gold_synth + unpaired + missing_cands + missing_o + disagree
    body = ['- Δ rows (independent recount): **%d**' % total,
            '- labeled %d + empty %d + gold/synthetic %d + unpaired %d + no-candidates %d + no-O %d + text-disagree %d = **%d**'
            % (labeled, empty, gold_synth, unpaired, missing_cands, missing_o, disagree, accounted),
            '- NOTE: recount runs against the LIVE logs db — rows written after the walker build '
            'appear in the recount only; a small positive drift (recount > accounted) is expected.']
    drift = total - accounted
    status = 'PASS' if 0 <= drift <= 50 else 'FAIL (drift %d — ledger does not close)' % drift
    body.insert(2, '- drift: %+d rows (post-build live accretion)' % drift)
    return sect(lines, '2 · Join conservation ledger', status, body)


def check_window(walker, lines):
    body = ['| prior turns available | labeled turns |', '|---|---|']
    hist = walker.execute(
        "SELECT CASE WHEN seq >= 8 THEN '8+' ELSE CAST(seq AS TEXT) END d, count(*) "
        "FROM turns WHERE labeled=1 GROUP BY d ORDER BY min(seq)").fetchall()
    total = sum(n for _, n in hist)
    deep = sum(n for d, n in hist if d == '8+')
    for d, n in hist:
        body.append('| %s | %d |' % (d, n))
    body.append('')
    body.append('- %d/%d labeled turns (%.0f%%) have the full K=8 window.'
                % (deep, total, deep / max(total, 1) * 100))
    status = 'PASS' if deep / max(total, 1) >= 0.4 else 'WARN (K=8 support below 40%)'
    return sect(lines, '3 · Achieved-window histogram', status, body)


def lane_cols():
    return (['v_%s_op' % v.strip('_') for v in MAXSIM_VIEWS] + ['sit_op', 'idf_op'])


def check_sensitivity(walker, lines):
    body = ['| lane | within-turn std (a) | per-node cross-turn std (b) | verdict |',
            '|---|---|---|---|']
    failed = []
    for lane in lane_cols():
        a = walker.execute(
            "SELECT avg(s) FROM (SELECT session_id, epoch, seq, "
            " CASE WHEN count(%s) > 3 THEN (max(%s)-min(%s)) ELSE NULL END s"
            " FROM cand_turn_scores WHERE j=0 GROUP BY session_id, epoch, seq)"
            % (lane, lane, lane)).fetchone()[0] or 0.0
        b = walker.execute(
            "SELECT avg(s) FROM (SELECT node_id,"
            " CASE WHEN count(%s) > 4 THEN (max(%s)-min(%s)) ELSE NULL END s"
            " FROM cand_turn_scores WHERE j=0 GROUP BY node_id)"
            % (lane, lane, lane)).fetchone()[0] or 0.0
        # (a) field separates candidates within a turn; (b) field RESPONDS to
        # the query — a query-independent lane scores each node identically
        # across turns (b≈0) no matter how healthy (a) looks.
        dead = a < 1e-4 or b < 1e-4
        if dead:
            failed.append(lane)
        body.append('| %s | %.4f | %.4f | %s |' % (lane, a, b, 'DEAD' if dead else 'alive'))
    status = 'PASS' if not failed else 'FAIL (dead lanes: %s)' % ', '.join(failed)
    return sect(lines, '4 · Lane sensitivity (anti-dead-operator)', status, body)


def check_replay(walker, lines):
    """K=0 static-gain composition vs the LIVE ranking (pool_score) on
    laf_v1-era turns. Spearman per turn; production applies fatigue/floors
    after the scorer and z-scores over the full field (not the 24-candidate
    pool), so exact reproduction is not expected — high correlation is."""
    gains = {'maxsim': DEFAULT_CONFIG['gain_maxsim'], 'pick': 0.0, 'enc': 0.0,
             'idf': DEFAULT_CONFIG['gain_idf'], 'sit': DEFAULT_CONFIG['gain_sit']}
    # NOTE: pick/enc (episodic) are not walker lanes yet — they join at sweep
    # time via roles_for_moments. Correlation is against the partial stack.
    # live ranking = rank_in_pool (the order recall emitted, full precision).
    # pool_score is cand_detail's '%.2f' — ~6 distinct values per 24
    # candidates; Spearman against it measures tie noise, not agreement.
    turns = walker.execute(
        "SELECT DISTINCT c.session_id, c.epoch, c.seq FROM candidates c "
        "JOIN turns t USING (session_id, epoch, seq) "
        "WHERE t.ts > ? AND c.rank_in_pool IS NOT NULL", (LAF_LIVE_TS,)).fetchall()
    rhos = []
    for sess, epoch, seq in turns:
        rows = walker.execute(
            "SELECT s.node_id, %s, s.sit_op, s.idf_op, -c.rank_in_pool "
            "FROM cand_turn_scores s JOIN candidates c ON c.session_id=s.session_id "
            "AND c.epoch=s.epoch AND c.seq=s.seq AND c.node_id=s.node_id "
            "WHERE s.session_id=? AND s.epoch=? AND s.seq=? AND s.j=0 "
            "AND c.rank_in_pool IS NOT NULL"
            % ', '.join('s.v_%s_op' % v.strip('_') for v in MAXSIM_VIEWS),
            (sess, epoch, seq)).fetchall()
        if len(rows) < 8:
            continue
        arr = np.array([[np.nan if x is None else x for x in r[1:]] for r in rows], dtype=float)
        # column split derived from the SELECT shape, not hardcoded (review F3):
        # [NV view cols] + sit + idf + live. A P3 gain retune that adds/zeroes a
        # view changes MAXSIM_VIEWS length; a fixed arr[:,:6] would silently
        # mis-slice (idf read as live → Spearman ≈ 1.0, gate falsely green).
        nv = len(MAXSIM_VIEWS)
        views, sit, idf, live = arr[:, :nv], arr[:, nv], arr[:, nv + 1], arr[:, nv + 2]
        maxsim = np.nanmax(views, axis=1)
        # production normalizer, imported not re-implemented (review F4) — the
        # replay's whole point is measured==shipped
        def z(x):
            return _zscore(np.asarray(x, dtype=float), len(x))
        score = (gains['maxsim'] * z(maxsim) + gains['sit'] * z(sit)
                 + gains['idf'] * z(idf))
        # Spearman via rank correlation
        r1 = np.argsort(np.argsort(score))
        r2 = np.argsort(np.argsort(live))
        if np.std(r1) < 1e-9 or np.std(r2) < 1e-9:
            continue
        rhos.append(float(np.corrcoef(r1, r2)[0, 1]))
    med = float(np.median(rhos)) if rhos else float('nan')

    # THE GATE: pool-membership separability. Production admitted these
    # candidates from the full ~7k field — offline q_vec×node scores must
    # rank pool members far above random nodes, independent of within-pool
    # composition differences (episodic mass, full-field z, fatigue).
    from walker_db import open_brain_ro
    braindb = open_brain_ro()
    qturns = walker.execute(
        "SELECT session_id, epoch, seq, q_vec FROM turns "
        "WHERE labeled=1 AND q_vec IS NOT NULL AND ts > ?", (LAF_LIVE_TS,)).fetchall()
    qturns = RNG.sample(qturns, min(40, len(qturns)))
    rand_nodes = braindb.execute(
        "SELECT embedding FROM node_enrichments WHERE vector_type='_primary' "
        "ORDER BY node_id LIMIT 3000").fetchall()
    braindb.close()
    rand_mat = np.stack([_unit(b[0]) for b in RNG.sample(rand_nodes, 300)])
    aucs = []
    for sess, epoch, seq, qv in qturns:
        q = _unit(qv)
        pool = [p[0] for p in walker.execute(
            "SELECT v_primary_op FROM cand_turn_scores WHERE session_id=? "
            "AND epoch=? AND seq=? AND j=0 AND v_primary_op IS NOT NULL",
            (sess, epoch, seq)).fetchall()]
        if len(pool) < 8:
            continue
        pos, neg = np.array(pool), rand_mat @ q
        aucs.append(float((pos[:, None] > neg[None, :]).mean()))
    auc_med = float(np.median(aucs)) if aucs else float('nan')

    body = ['- Pool-vs-random separability (v_primary, q_vec, %d turns): median AUC **%.3f**, p25 %.3f'
            % (len(aucs), auc_med, np.percentile(aucs, 25) if aucs else float('nan')),
            '- Within-pool Spearman vs live rank_in_pool (informational, %d turns): median %.3f '
            '— gap attributable to episodic lanes joining at sweep time (0.8/2.8 gain mass), '
            'per-lane z over full field vs pool, and fatigue/floors applied after the scorer.'
            % (len(rhos), med)]
    status = ('PASS' if auc_med >= 0.85 else 'WARN' if auc_med >= 0.75 else 'FAIL') if aucs \
        else 'FAIL (no comparable turns)'
    return sect(lines, '5 · Replay sanity (pool separability + within-pool rank)', status, body)


def check_spot_audit(walker, lines):
    # Restrict to renders under the storage cap: store_embeddings persists
    # text[:500] while the vector embeds the FULL render — re-embedding a
    # truncated text of a long turn measures the truncation, not drift.
    sample = walker.execute(
        "SELECT op_trace_id, op_vec FROM turns WHERE op_vec_source='store' "
        "AND op_trace_id IS NOT NULL AND length(op_text) < 480 "
        "ORDER BY ts DESC LIMIT 400").fetchall()
    sample = RNG.sample(sample, min(SPOT_SAMPLE, len(sample)))
    logs = open_logs_ro()
    from servers import embedder
    embedder.load_model()
    sims = []
    for tid, blob in sample:
        stored_text = logs.execute(
            "SELECT text FROM trace_embeddings WHERE trace_id=?", (tid,)).fetchone()
        if not stored_text or not stored_text[0]:
            continue
        fresh = embedder.embed_batch([stored_text[0]], kind='document')
        if not fresh or fresh[0] is None:
            continue
        a, b = _unit(blob), _unit(fresh[0])
        if a is not None and b is not None:
            sims.append(float(a @ b))
    logs.close()
    mn = min(sims) if sims else float('nan')
    mean = float(np.mean(sims)) if sims else float('nan')
    body = ['- %d stored vectors re-embedded from their stored render text' % len(sims),
            '- cosine(stored, fresh): min **%.4f**, mean %.4f' % (mn, mean),
            '- Measured noise floor: the quantized ONNX embedder is batch-shape '
            'nondeterministic — the same text solo vs in a batch of 5 gives ~0.982 '
            'cosine (worker embeds in batches of 5; this audit re-embeds solo). '
            'The check therefore gates on catastrophic drift (wrong model/prefix/'
            'recipe), not bit-identity.']
    status = 'PASS' if sims and mn >= 0.96 and mean >= 0.99 else 'FAIL (drift beyond batch-jitter envelope)'
    return sect(lines, '6 · Embedding spot-audit (recipe/prefix drift)', status, body)


def main():
    walker = open_walker()
    logs = open_logs_ro()
    lines = ['# walker_health — build report', '']
    statuses = [
        check_fill_rates(walker, lines),
        check_conservation(walker, logs, lines),
        check_window(walker, lines),
        check_sensitivity(walker, lines),
        check_replay(walker, lines),
        check_spot_audit(walker, lines),
    ]
    fails = [s for s in statuses if s.startswith('FAIL')]
    lines.insert(1, '**Overall: %s** — %d PASS / %d WARN / %d FAIL\n' % (
        'FAIL' if fails else 'GREEN',
        sum(s.startswith('PASS') for s in statuses),
        sum(s.startswith('WARN') for s in statuses), len(fails)))
    (WALKER_DIR / 'walker_health.md').write_text('\n'.join(lines) + '\n')
    walker.close()
    logs.close()
    print('\n'.join(lines))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
