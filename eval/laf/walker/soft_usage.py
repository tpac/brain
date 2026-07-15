"""Walker phase 4 — soft-usage label (§20.12 A4, H2 pre-run amendment) + audit.

The registered used_next_1/3 hard label is statistically dead (~5-10
positives / 110k rows). Its replacement as the debias leg: SOFT USAGE —
similarity between each surfaced candidate and Anchor's ACTUAL next
response. High soft-usage = the memory's content showed up in what Anchor
actually said next; the graded signal picks/drops can't give.

Response resolution (F2-safe by construction — keyed by seq, bounded by the
stop, no forward windows):
    response(turn s) = the turn's OWN attached assistant message if present
                       (84% of labeled turns; walker v4 attaches the stop's
                       assistant_message to the LAST s0 of the stop, so
                       last-of-stop turns own theirs);
                     = else the anchor of the FIRST anchored turn AFTER s in
                       the same (session, epoch, stop) — the assistant
                       message that actually followed a superseded prompt
                       (its successor closed the stop). Multi-anchor stops
                       exist (/watch wake micro-turns each carry a tiny
                       '(watching…)' message), which is why min-seq-after,
                       never 'the stop's anchor';
                     = else NULL (session tails; ledger-counted).

Label values, per (labeled turn, candidate):
    soft_max  = nanmax over the 6 content-view cosines vs the response vec
                (PRIMARY — same aggregation as the engine's maxsim lane)
    soft_mean = nanmean over the same 6 (SECONDARY, reported side-by-side;
                the nanmax-enrichment bias is a known maxsim critique)
sit/idf views are deliberately excluded — usage is content similarity.
own-anchor rows reuse the walker's stored v_*_anchor j=0 columns (the exact
values the cross-check just proved against the engine); stop-resolved rows
are computed fresh from the same stored vectors with the same _unit/matmul.

AUDIT (runs after the build, writes soft_usage.md) — the label gates
NOTHING until this passes, per A4:
  1. picks: AUC(soft_max: selected vs dropped), overall + per response
     source + per era (pre/post 2026-06-08 trace-loss boundary).
     PRE-DECLARED BAR: overall AUC > 0.55, else the label FAILS and the
     sweep's debias leg reverts to picks + gold only.
  2. gold: at the 24 gold-cue turns present in the walker, gold-tier nodes'
     median soft_max vs the turn's other candidates (count of cues where
     gold wins; sparse — reported, not gated).
  3. distribution sanity: percentiles + NULL ledger (a degenerate label
     passes no audit).

Run:  ./dev python3 eval/laf/walker/soft_usage.py [--rebuild]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import (open_walker, open_brain_ro, EXTRACT_VERSION,
                       EMBED_VERSION, lanes_version, WALKER_DIR)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import MAXSIM_VIEWS, _unit          # noqa: E402
from scores import load_node_vectors                        # noqa: E402

SOFT_USAGE_VERSION = 'v1-j0anchor-minseqafter|' + ','.join(
    v.strip('_') for v in MAXSIM_VIEWS)
AUC_BAR = 0.55                    # pre-declared; below this the label fails
ERA_SPLIT = '2026-06-08'          # trace-loss boundary (§20.12 A6)
REPORT = WALKER_DIR / 'soft_usage.md'
ANCHOR_COLS = ['v_%s_anchor' % v.strip('_') for v in MAXSIM_VIEWS]


def gate_provenance(walker):
    stamps = dict(walker.execute(
        "SELECT key, value FROM build_meta WHERE key IN "
        "('extract_version','embed_version','scores_lanes_version')"))
    expect = {'extract_version': EXTRACT_VERSION,
              'embed_version': EMBED_VERSION,
              'scores_lanes_version': lanes_version(MAXSIM_VIEWS)}
    bad = {k: (stamps.get(k), v) for k, v in expect.items()
           if stamps.get(k) != v}
    if bad:
        raise SystemExit('soft_usage: stale walker artifact — %s. Rebuild '
                         '(extract → embed → scores), never bypass.'
                         % ', '.join('%s stamped %r expects %r' % (k, a, b)
                                     for k, (a, b) in bad.items()))


def resolve_responses(walker, c):
    """{(sess, epoch, seq): ('own'|'stop_resolved', anchor_vec_blob)} for
    every labeled turn (absent = unresolvable, NULL label)."""
    anchored = defaultdict(list)      # (sess, epoch, stop) → [(seq, blob)]
    labeled = []                      # (sess, epoch, seq, stop, own_blob)
    for sess, epoch, seq, stop, labeled_f, av in walker.execute(
            "SELECT session_id, epoch, seq, stop, labeled, anchor_vec "
            "FROM turns"):
        if av is not None:
            anchored[(sess, epoch, stop)].append((seq, av))
        if labeled_f:
            labeled.append((sess, epoch, seq, stop, av))
    out = {}
    for sess, epoch, seq, stop, own in labeled:
        if own is not None:
            out[(sess, epoch, seq)] = ('own', own)
            c['resp_own'] += 1
            continue
        after = sorted((s, b) for s, b in anchored.get((sess, epoch, stop),
                                                       []) if s > seq)
        if after:
            out[(sess, epoch, seq)] = ('stop_resolved', after[0][1])
            c['resp_stop_resolved'] += 1
        else:
            c['resp_unresolved'] += 1
    return out


def build(walker, c):
    walker.executescript(
        'CREATE TABLE IF NOT EXISTS soft_usage ('
        ' session_id TEXT NOT NULL, epoch INTEGER NOT NULL,'
        ' seq INTEGER NOT NULL, node_id TEXT NOT NULL,'
        ' soft_max REAL, soft_mean REAL, source TEXT NOT NULL,'
        ' PRIMARY KEY (session_id, epoch, seq, node_id))')
    responses = resolve_responses(walker, c)

    cand_by_turn = defaultdict(list)
    for sess, epoch, seq, nid in walker.execute(
            "SELECT c.session_id, c.epoch, c.seq, c.node_id FROM candidates c "
            "JOIN turns t ON t.session_id=c.session_id AND t.epoch=c.epoch "
            " AND t.seq=c.seq WHERE t.labeled=1 AND c.node_id IS NOT NULL"):
        cand_by_turn[(sess, epoch, seq)].append(nid)

    # own-anchor rows: the stored j=0 anchor columns ARE the label inputs
    stored = {}
    for row in walker.execute(
            "SELECT session_id, epoch, seq, node_id, %s "
            "FROM cand_turn_scores WHERE j=0" % ', '.join(ANCHOR_COLS)):
        stored[(row[0], row[1], row[2], row[3])] = row[4:]

    # stop-resolved rows need fresh cosines from the same stored vectors
    resolved_turns = [k for k, (src, _) in responses.items()
                      if src == 'stop_resolved']
    need_nodes = {nid for k in resolved_turns for nid in cand_by_turn.get(k, [])}
    node_vecs = {}
    if need_nodes:
        braindb = open_brain_ro()
        node_vecs = load_node_vectors(braindb, need_nodes)
        braindb.close()

    rows = []
    for key, cands in sorted(cand_by_turn.items()):
        resp = responses.get(key)
        for nid in cands:
            if resp is None:
                rows.append((*key, nid, None, None, 'unresolved'))
                c['rows_unresolved'] += 1
                continue
            src, blob = resp
            if src == 'own':
                vals = stored.get((*key, nid))
                sims = (np.array([np.nan if v is None else float(v)
                                  for v in vals], dtype=float)
                        if vals is not None else np.full(len(ANCHOR_COLS),
                                                         np.nan))
            else:
                rv = _unit(blob)
                sims = np.array(
                    [float(node_vecs[v][nid] @ rv)
                     if rv is not None and nid in node_vecs.get(v, {})
                     else np.nan for v in MAXSIM_VIEWS], dtype=float)
            if np.all(np.isnan(sims)):
                rows.append((*key, nid, None, None, src + '_novec'))
                c['rows_no_vectors'] += 1
            else:
                with np.errstate(all='ignore'):
                    rows.append((*key, nid, float(np.nanmax(sims)),
                                 float(np.nanmean(sims)), src))
                c['rows_labeled'] += 1
    walker.executemany(
        'INSERT OR REPLACE INTO soft_usage (session_id, epoch, seq, node_id,'
        ' soft_max, soft_mean, source) VALUES (?,?,?,?,?,?,?)', rows)
    walker.executemany(
        "INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
        [('soft_usage_version', SOFT_USAGE_VERSION)] +
        [('soft_' + k, str(v)) for k, v in sorted(c.items())])
    walker.commit()


def auc(pos, neg):
    """Mann-Whitney AUC — P(random positive > random negative)."""
    if not len(pos) or not len(neg):
        return None
    both = np.concatenate([pos, neg])
    ranks = both.argsort().argsort().astype(float) + 1.0
    # midranks for ties
    order = both.argsort()
    sb = both[order]
    i = 0
    while i < len(sb):
        j = i
        while j + 1 < len(sb) and sb[j + 1] == sb[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def audit(walker, lines):
    rows = walker.execute(
        "SELECT s.soft_max, s.soft_mean, s.source, c.outcome, t.ts, s.node_id,"
        " s.session_id, t.stop "
        "FROM soft_usage s "
        "JOIN candidates c ON c.session_id=s.session_id AND c.epoch=s.epoch"
        " AND c.seq=s.seq AND c.node_id=s.node_id "
        "JOIN turns t ON t.session_id=s.session_id AND t.epoch=s.epoch"
        " AND t.seq=s.seq").fetchall()
    ok = [r for r in rows if r[0] is not None]
    lines.append('## 1 · picks correlation (AUC selected-vs-dropped)')
    lines.append('')
    lines.append('| slice | n_sel | n_drop | AUC soft_max | AUC soft_mean |')
    lines.append('|---|---|---|---|---|')
    slices = {
        'ALL': lambda r: True,
        'own-anchor': lambda r: r[2] == 'own',
        'stop-resolved': lambda r: r[2] == 'stop_resolved',
        'era pre-%s' % ERA_SPLIT: lambda r: (r[4] or '') < ERA_SPLIT,
        'era post-%s' % ERA_SPLIT: lambda r: (r[4] or '') >= ERA_SPLIT,
    }
    overall_auc = None
    for name, f in slices.items():
        sel = [r for r in ok if f(r) and r[3] == 'selected']
        drp = [r for r in ok if f(r) and r[3] == 'dropped']
        a_max = auc(np.array([r[0] for r in sel]),
                    np.array([r[0] for r in drp]))
        a_mean = auc(np.array([r[1] for r in sel]),
                     np.array([r[1] for r in drp]))
        if name == 'ALL':
            overall_auc = a_max
        lines.append('| %s | %d | %d | %s | %s |'
                     % (name, len(sel), len(drp),
                        '%.4f' % a_max if a_max is not None else '—',
                        '%.4f' % a_mean if a_mean is not None else '—'))
    lines.append('')

    # gold agreement (sparse; reported, not gated)
    manifest = json.loads((WALKER_DIR / 'gold_manifest.json').read_text())
    gold = json.loads(
        (WALKER_DIR.parents[2] / 'eval' / 'oracle_audit' / 'gold_remint'
         / 'frozen_gold_24.json').read_text())
    by_turn = defaultdict(list)
    for r in ok:
        by_turn[(r[6], r[7])].append(r)
    lines.append('## 2 · gold agreement (median soft_max, gold vs rest)')
    lines.append('')
    wins = comparable = 0
    for cue in manifest['cues']:
        if not cue.get('matched'):
            continue
        shorts = {g['node_id'] for t in ('gold_plus', 'gold')
                  for g in gold[cue['cue_id']]['tiers'][t]}
        turn_rows = by_turn.get((cue['session_id'], cue['stop']), [])
        g = [r[0] for r in turn_rows if r[5][:8] in shorts]
        rest = [r[0] for r in turn_rows if r[5][:8] not in shorts]
        if g and rest:
            comparable += 1
            if np.median(g) > np.median(rest):
                wins += 1
    lines.append('- comparable cue-turns (gold node among labeled rows): %d'
                 % comparable)
    lines.append('- gold-median beats rest-median: %d/%d' % (wins, comparable))
    if comparable == 0:
        excl = walker.execute("SELECT value FROM build_meta WHERE "
                              "key='extract_sessions_gold_excluded'").fetchone()
        lines.append('- NOTE: 0 is BY CONSTRUCTION, not a broken join — the '
                     'walker excludes all gold-cue sessions at extraction '
                     '(anti-leak; ledger: extract_sessions_gold_excluded=%s). '
                     '"Correlate with gold where both exist" (A4) has no '
                     'population: both never exist. The gold check lives in '
                     'the sweep\'s reach-Δ leg instead.'
                     % (excl[0] if excl else '?'))
    lines.append('')

    vals = np.array([r[0] for r in ok])
    lines.append('## 3 · distribution sanity')
    lines.append('')
    lines.append('- labeled rows: %d; NULL rows: %d' % (len(ok),
                                                        len(rows) - len(ok)))
    lines.append('- soft_max percentiles p1/p25/p50/p75/p99: '
                 + '/'.join('%.3f' % np.percentile(vals, p)
                            for p in (1, 25, 50, 75, 99)))
    lines.append('- std: %.4f' % float(vals.std()))
    lines.append('')
    verdict = ('PASS' if overall_auc is not None and overall_auc > AUC_BAR
               else 'FAIL — sweep debias leg reverts to picks + gold only')
    lines.append('**Pre-declared bar: AUC(ALL, soft_max) > %.2f → %s '
                 '(measured %.4f)**'
                 % (AUC_BAR, verdict,
                    overall_auc if overall_auc is not None else float('nan')))
    return verdict


def main():
    rebuild = '--rebuild' in sys.argv
    walker = open_walker()
    gate_provenance(walker)
    if rebuild:
        walker.execute('DROP TABLE IF EXISTS soft_usage')
        walker.commit()
    have = walker.execute(
        "SELECT value FROM build_meta WHERE key='soft_usage_version'"
    ).fetchone()
    c = defaultdict(int)
    if have and have[0] == SOFT_USAGE_VERSION and not rebuild:
        print('soft_usage table current (%s) — audit only' % have[0])
    elif have and have[0] != SOFT_USAGE_VERSION and not rebuild:
        raise SystemExit('soft_usage stamped %s, code is %s — rerun with '
                         '--rebuild' % (have[0], SOFT_USAGE_VERSION))
    else:
        build(walker, c)
        print('build counters:')
        for k in sorted(c):
            print('  %-24s %d' % (k, c[k]))
    lines = ['# soft_usage — label build + quality audit (§20.12 A4)', '',
             'version: %s' % SOFT_USAGE_VERSION, '']
    verdict = audit(walker, lines)
    walker.close()
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
