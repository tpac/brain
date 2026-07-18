"""P4 SUMMARY.md — the moment-stack A/B in the compare_arms house style.

Renders one report per pooled corpus from the Leg A/B artifacts already on
disk (leg_b/*.jsonl, walker soft labels, external_score.json): headline
table, decision criteria with thresholds, per-item side-by-side with
movement marks. Baseline arm = A0 (production), candidate = A1 (moment
table); A0f / A1a / C1 ride as reference columns.

USE
    ./dev python3 eval/longmem/p4_report.py --corpus 74aea3
    → <corpus_dir>/leg_b/SUMMARY.md (+ printed)
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus import corpus_dir  # noqa: E402

ARMS = ('A0', 'A0f', 'A1', 'A1a', 'C1')
BASE, CAND = 'A0', 'A1'
_SID = re.compile(r'^i[0-9a-f]{7}-(.+)-s\d+$')


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 3 else float('nan')


def qid_of(sid):
    m = _SID.match(sid)
    return m.group(1) if m else sid


def load(corpus_hash):
    cdir = Path(corpus_dir(corpus_hash))
    manifest = json.loads((cdir / 'manifest.json').read_text())
    conn = sqlite3.connect(
        'file:%s?mode=ro' % (cdir / 'walker' / 'walker.db'), uri=True)
    soft = {(r[0], r[1], r[2], r[3]): r[4] for r in conn.execute(
        'SELECT session_id, epoch, seq, node_id, soft_max FROM soft_usage '
        'WHERE soft_max IS NOT NULL')}
    runs = {}
    for arm in ARMS:
        f = cdir / 'leg_b' / ('%s.jsonl' % arm)
        if f.exists():
            runs[arm] = [json.loads(l) for l in f.read_text().splitlines() if l]
    report_json = cdir / 'leg_b' / 'leg_b_report.json'
    leg_b = json.loads(report_json.read_text()) if report_json.exists() else {}
    return cdir, manifest, soft, runs, leg_b


def arm_stats(rows, soft):
    """Per-arm pooled soft_r + mean delivered soft@5 + per-session xs/ys."""
    per_sess = defaultdict(lambda: ([], []))
    top5 = []
    ms = []
    for r in rows:
        key = tuple(r['key'])
        ms.append(r['ms'])
        sms = [(nid, soft.get((*key, nid)), s) for nid, s in r['cands']]
        t5 = [sm for _, sm, _ in sms[:5] if sm is not None]
        if t5:
            top5.append(float(np.mean(t5)))
        xs, ys = per_sess[key[0]]
        for _, sm, s in sms:
            if sm is not None and s is not None:
                xs.append(s)
                ys.append(sm)
    pooled = pearson(*(np.concatenate([v[i] for v in per_sess.values()])
                       for i in (0, 1)))
    return {'per_sess': dict(per_sess), 'soft_r': pooled,
            'soft_at5': float(np.mean(top5)),
            'p50': float(np.percentile(ms, 50)),
            'p95': float(np.percentile(ms, 95))}


def clustered_ci(a_sess, b_sess, n=2000, seed=42):
    sess = sorted(set(a_sess) & set(b_sess))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(sess), len(sess), replace=True)
        xa, ya, xb, yb = [], [], [], []
        for i in pick:
            s = sess[i]
            xa += a_sess[s][0]; ya += a_sess[s][1]
            xb += b_sess[s][0]; yb += b_sess[s][1]
        diffs.append(pearson(xa, ya) - pearson(xb, yb))
    return [float(x) for x in np.nanpercentile(diffs, [2.5, 97.5])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    args = p.parse_args()

    cdir, manifest, soft, runs, leg_b = load(args.corpus)
    axis_of = {it['qid']: it['axis'] for it in manifest['items']}
    stats = {a: arm_stats(rows, soft) for a, rows in runs.items()}

    ci = clustered_ci(stats[CAND]['per_sess'], stats['A0f']['per_sess'])
    delta = stats[CAND]['soft_r'] - stats['A0f']['soft_r']
    c2 = leg_b.get('c2') or {}

    # per-qid: mean delivered soft@5 per arm (sessions of the item pooled)
    per_qid = defaultdict(lambda: defaultdict(list))
    for arm, rows in runs.items():
        for r in rows:
            key = tuple(r['key'])
            t5 = [soft[(*key, nid)] for nid, _ in r['cands'][:5]
                  if (*key, nid) in soft]
            if t5:
                per_qid[qid_of(key[0])][arm].append(float(np.mean(t5)))

    L = []
    L.append('# P4 A/B — `A0 (production)` vs `A1 (moment table)` — '
             'corpus `%s` (%s)' % (args.corpus, manifest['label']))
    L.append('')
    L.append('**Arm A (baseline):** production gains, no moment stack')
    L.append('**Arm B (candidate):** frozen definitive-fit table, K=8 '
             '(references: A0f fitted-j0 · A1a additive · C1 shuffle control)')
    L.append('')
    L.append('Instrument: judge-free soft-usage vs the benchmark\'s own next '
             'assistant response; real `brain.recall` per cue with as_of '
             '(%d cues). True-accuracy QA leg rides sweep/compare_arms '
             'separately.' % leg_b.get('n_cues', 0))
    L.append('')
    L.append('## Headline')
    L.append('')
    L.append('| | A0 | A1 | Δ | A0f | A1a | C1 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    L.append('| soft_r (pooled) | %.3f | %.3f | %+.3f | %.3f | %.3f | %.3f |'
             % (stats['A0']['soft_r'], stats['A1']['soft_r'],
                stats['A1']['soft_r'] - stats['A0']['soft_r'],
                stats['A0f']['soft_r'], stats['A1a']['soft_r'],
                stats['C1']['soft_r']))
    L.append('| delivered soft@5 | %.3f | %.3f | %+.3f | %.3f | %.3f | %.3f |'
             % (stats['A0']['soft_at5'], stats['A1']['soft_at5'],
                stats['A1']['soft_at5'] - stats['A0']['soft_at5'],
                stats['A0f']['soft_at5'], stats['A1a']['soft_at5'],
                stats['C1']['soft_at5']))
    L.append('| recall p50 (ms) | %.0f | %.0f | %+.0f | %.0f | %.0f | %.0f |'
             % (stats['A0']['p50'], stats['A1']['p50'],
                stats['A1']['p50'] - stats['A0']['p50'],
                stats['A0f']['p50'], stats['A1a']['p50'], stats['C1']['p50']))
    L.append('| recall p95 (ms) | %.0f | %.0f | %+.0f | %.0f | %.0f | %.0f |'
             % (stats['A0']['p95'], stats['A1']['p95'],
                stats['A1']['p95'] - stats['A0']['p95'],
                stats['A0f']['p95'], stats['A1a']['p95'], stats['C1']['p95']))
    L.append('')
    L.append('## Decision criteria (§20.18, pre-committed)')
    L.append('')
    L.append('| Criterion | Threshold | Measured | Pass? |')
    L.append('|---|---|---|:---:|')
    p_pass = delta >= 0.05 and ci[0] > 0
    L.append('| P-primary | Δ(A1−A0f) ≥ +0.05, clustered CI > 0 | '
             '%+.3f, CI [%.3f, %.3f] | %s |'
             % (delta, ci[0], ci[1], '✓' if p_pass else '✗'))
    c1_pass = stats['C1']['soft_r'] < stats['A0f']['soft_r']
    L.append('| C1 shuffle | donor history must NOT beat A0f | '
             '%.3f vs %.3f | %s |'
             % (stats['C1']['soft_r'], stats['A0f']['soft_r'],
                '✓' if c1_pass else '✗'))
    L.append('| C2 identity | empty-stack A1 ≡ A0f bit-identical | '
             '%s mismatches / %s cues | %s |'
             % (len(c2.get('mismatches', [])), c2.get('first_cues', '?'),
                '✓' if c2.get('pass') else '✗'))
    lat_pass = stats['A1']['p95'] <= 1.5 * stats['A0']['p95']
    L.append('| Latency | A1 p95 ≤ 1.5× A0 p95 | %.0fms vs %.0fms | %s |'
             % (stats['A1']['p95'], stats['A0']['p95'],
                '✓' if lat_pass else '✗'))
    L.append('')
    L.append('## Per-item side by side (mean delivered soft@5)')
    L.append('')
    L.append('| qid | axis | A0 | A1 | Δ | move |')
    L.append('|---|---|---:|---:|---:|:---:|')
    for qid in sorted(per_qid):
        arms = per_qid[qid]
        a = float(np.mean(arms.get('A0', [np.nan])))
        b = float(np.mean(arms.get('A1', [np.nan])))
        d = b - a
        move = '↑' if d > 0.01 else ('↓' if d < -0.01 else '=')
        L.append('| `%s` | %s | %.3f | %.3f | %+.3f | %s |'
                 % (qid, axis_of.get(qid, '?'), a, b, d, move))
    ups = sum(1 for q in per_qid
              if np.mean(per_qid[q].get('A1', [0])) -
              np.mean(per_qid[q].get('A0', [0])) > 0.01)
    downs = sum(1 for q in per_qid
                if np.mean(per_qid[q].get('A1', [0])) -
                np.mean(per_qid[q].get('A0', [0])) < -0.01)
    L.append('')
    L.append('Movement: **%d ↑ / %d ↓ / %d =** of %d items.'
             % (ups, downs, len(per_qid) - ups - downs, len(per_qid)))

    out = cdir / 'leg_b' / 'SUMMARY.md'
    out.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nreport → %s' % out)


if __name__ == '__main__':
    main()
