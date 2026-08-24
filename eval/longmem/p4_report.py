"""P4 SUMMARY.md — the moment-stack A/B in the compare_arms house style.

Three-level KPI pyramid over the Leg B engine runs:

  L1 headline   paired win/tie/loss (per-cue delivered top-5 usage),
                target-hit@5 strict AND twin-credited, per arm.
  L2 mechanism  where the delta lives: reach-vs-rerank split, depth cells,
                cue-length cells, pivot cells (within-session relative
                split), controls as cells.
  L3 stats      pooled soft_r + clustered CI (the §20.18 P-primary),
                latency, coverage ledger.

Robustness rules (2026-07-18 miss-taxonomy review):
  - target eligibility excludes seed-pack nodes (anchor:seed) — label noise;
  - twin-credited hit: a delivered node whose TITLE cosine ≥ 0.85 with the
    target counts (consolidation debt is a substrate problem, not an arm
    difference — strict is reported alongside).

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
TARGET_SOFT = 0.70
TWIN_COS = 0.85
WIN_EPS = 0.02
_SID = re.compile(r'^i[0-9a-f]{7}-(.+)-s\d+$')
_DATE = re.compile(r'^\[Current date:[^\]]*\]\s*')


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 3 else float('nan')


def unit(blob):
    v = np.frombuffer(blob, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


class Data:
    def __init__(self, corpus_hash):
        self.cdir = Path(corpus_dir(corpus_hash))
        self.manifest = json.loads((self.cdir / 'manifest.json').read_text())
        walker = sqlite3.connect(
            'file:%s?mode=ro' % (self.cdir / 'walker' / 'walker.db'), uri=True)
        brain = sqlite3.connect(
            'file:%s?mode=ro' % (self.cdir / 'pooled' / 'brain.db'), uri=True)
        self.soft = {(r[0], r[1], r[2], r[3]): r[4] for r in walker.execute(
            'SELECT session_id, epoch, seq, node_id, soft_max FROM soft_usage '
            'WHERE soft_max IS NOT NULL')}
        self.meta = {(r[0], r[1], r[2]): (len(_DATE.sub('', r[3] or '')),)
                     for r in walker.execute(
                         'SELECT session_id, epoch, seq, op_text FROM turns')}
        self.opvec = {}
        for sid, ep, seq, blob in walker.execute(
                'SELECT session_id, epoch, seq, op_vec FROM turns '
                'WHERE op_vec IS NOT NULL'):
            self.opvec[(sid, ep, seq)] = unit(blob)
        self.seeds = {r[0] for r in brain.execute(
            "SELECT id FROM nodes WHERE encoding_source='anchor:seed'")}
        self.title_vec = {r[0]: unit(r[1]) for r in brain.execute(
            "SELECT node_id, embedding FROM node_enrichments "
            "WHERE vector_type='title' AND embedding IS NOT NULL")}
        self.runs = {}
        for arm in ARMS:
            f = self.cdir / 'leg_b' / ('%s.jsonl' % arm)
            if f.exists():
                self.runs[arm] = {tuple(r['key']): r for r in
                                  map(json.loads, f.read_text().splitlines())}
        rep = self.cdir / 'leg_b' / 'leg_b_report.json'
        self.leg_b = json.loads(rep.read_text()) if rep.exists() else {}
        self.keys = sorted(set.intersection(*[set(v) for v in self.runs.values()]))

    def target_of(self, key):
        """Highest-soft NON-SEED labeled node across all arms' candidates."""
        cands = set()
        for arm in self.runs:
            cands |= {n for n, _ in self.runs[arm][key]['cands']}
        lab = [(n, self.soft[(*key, n)]) for n in cands
               if (*key, n) in self.soft and n not in self.seeds]
        if not lab:
            return None, None
        n, s = max(lab, key=lambda x: x[1])
        return (n, s) if s >= TARGET_SOFT else (None, None)

    def hit(self, key, arm, target, credited):
        tv = self.title_vec.get(target)
        for n, _ in self.runs[arm][key]['cands'][:5]:
            if n == target:
                return True
            if credited and tv is not None:
                nv = self.title_vec.get(n)
                if nv is not None and float(tv @ nv) >= TWIN_COS:
                    return True
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    args = p.parse_args()
    d = Data(args.corpus)
    axis_of = {it['qid']: it['axis'] for it in d.manifest['items']}

    # ── per-cue paired win/tie/loss + latency + per-session soft_r rows ──
    win = tie = loss = 0
    per_sess = {a: defaultdict(lambda: ([], [])) for a in d.runs}
    lat = {a: [] for a in d.runs}
    for key in d.keys:
        m5 = {}
        for arm in d.runs:
            r = d.runs[arm][key]
            lat[arm].append(r['ms'])
            t5 = [d.soft[(*key, n)] for n, _ in r['cands'][:5]
                  if (*key, n) in d.soft]
            m5[arm] = np.mean(t5) if t5 else None
            xs, ys = per_sess[arm][key[0]]
            for n, s in r['cands']:
                sm = d.soft.get((*key, n))
                if sm is not None and s is not None:
                    xs.append(s)
                    ys.append(sm)
        if m5[BASE] is not None and m5[CAND] is not None:
            dd = m5[CAND] - m5[BASE]
            win += dd > WIN_EPS
            loss += dd < -WIN_EPS
            tie += abs(dd) <= WIN_EPS

    soft_r = {a: pearson(*(np.concatenate([v[i] for v in per_sess[a].values()])
                           for i in (0, 1))) for a in d.runs}

    # clustered bootstrap CI on r(A1) − r(A0f)
    sess = sorted(set(per_sess[CAND]) & set(per_sess['A0f']))
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(len(sess), len(sess), replace=True)
        xa, ya, xb, yb = [], [], [], []
        for i in pick:
            s = sess[i]
            xa += per_sess[CAND][s][0]; ya += per_sess[CAND][s][1]
            xb += per_sess['A0f'][s][0]; yb += per_sess['A0f'][s][1]
        diffs.append(pearson(xa, ya) - pearson(xb, yb))
    ci = [float(x) for x in np.nanpercentile(diffs, [2.5, 97.5])]
    delta = soft_r[CAND] - soft_r['A0f']

    # ── targets: strict + credited hits, cells ──────────────────────────
    hits = {a: {'strict': 0, 'cred': 0} for a in d.runs}
    n_t = 0
    gains = {'reach': 0, 'rerank': 0}
    losses = {'reach': 0, 'rerank': 0}
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # dim→bucket→[n,h0,h1]
    per_qid = defaultdict(lambda: defaultdict(list))
    sess_cos = defaultdict(list)      # within-session cos values for pivot split
    key_cos = {}
    for key in d.keys:
        v, pv = d.opvec.get(key), d.opvec.get((key[0], key[1], key[2] - 1))
        if v is not None and pv is not None:
            c = float(v @ pv)
            key_cos[key] = c
            sess_cos[key[0]].append(c)
    pivot_thr = {s: np.percentile(v, 25) for s, v in sess_cos.items() if len(v) >= 4}

    for key in d.keys:
        t, ts = d.target_of(key)
        # per-qid delivered usage rides every cue
        qid = (_SID.match(key[0]) or [None, key[0]]).__getitem__(1)
        for arm in (BASE, CAND):
            t5 = [d.soft[(*key, n)] for n, _ in d.runs[arm][key]['cands'][:5]
                  if (*key, n) in d.soft]
            if t5:
                per_qid[qid][arm].append(float(np.mean(t5)))
        if not t:
            continue
        n_t += 1
        h = {}
        for arm in d.runs:
            hits[arm]['strict'] += d.hit(key, arm, t, credited=False)
            hits[arm]['cred'] += d.hit(key, arm, t, credited=True)
            h[arm] = d.hit(key, arm, t, credited=True)
        h0, h1 = h[BASE], h[CAND]
        if h1 and not h0:
            in25 = any(n == t for n, _ in d.runs[BASE][key]['cands'])
            gains['rerank' if in25 else 'reach'] += 1
        if h0 and not h1:
            in25 = any(n == t for n, _ in d.runs[CAND][key]['cands'])
            losses['rerank' if in25 else 'reach'] += 1
        seq = key[2]
        db = 'seq0' if seq == 0 else ('seq1-3' if seq <= 3 else 'seq4+')
        oplen = d.meta.get(key, (0,))[0]
        lb = 'short' if oplen < 185 else ('mid' if oplen < 250 else 'long')
        cells = [('depth', db), ('cue-len', lb)]
        c = key_cos.get(key)
        thr = pivot_thr.get(key[0])
        if c is not None and thr is not None:
            cells.append(('pivot', 'pivot(q1)' if c <= thr else 'continuation'))
        for dim, b in cells:
            e = cell[dim][b]
            e[0] += 1
            e[1] += h0
            e[2] += h1

    # ── render ───────────────────────────────────────────────────────────
    L = ['# P4 A/B — `A0 (production)` vs `A1 (moment table)` — corpus '
         '`%s` (%s)' % (args.corpus, d.manifest['label']), '',
         '**Arm A (baseline):** production gains, no moment stack',
         '**Arm B (candidate):** frozen definitive-fit table, K=8 '
         '(references: A0f fitted-j0 · A1a additive · C1 shuffle control)', '',
         '%d cues · %d with a ≥%.2f non-seed target · twin-credit: title-cos '
         '≥ %.2f' % (len(d.keys), n_t, TARGET_SOFT, TWIN_COS), '',
         '## L1 — headline', '',
         '| | A0 | A1 | Δ | A0f | A1a | C1 |', '|---|---:|---:|---:|---:|---:|---:|']

    def row(name, f, fmt='%.0f%%', dfmt='%+.0fpp', scale=100):
        vals = {a: f(a) for a in ARMS if a in d.runs}
        L.append('| %s | %s | %s | %s | %s | %s | %s |' % (
            name, fmt % (scale * vals['A0']), fmt % (scale * vals['A1']),
            dfmt % (scale * (vals['A1'] - vals['A0'])),
            fmt % (scale * vals['A0f']), fmt % (scale * vals['A1a']),
            fmt % (scale * vals['C1'])))

    L.append('| paired win/tie/loss (per-cue top-5 usage) | — | **%d/%d/%d** '
             '| win-rate %.0f%% | — | — | — |'
             % (win, tie, loss, 100 * win / max(win + loss, 1)))
    row('target-hit@5 (twin-credited)', lambda a: hits[a]['cred'] / n_t)
    row('target-hit@5 (strict)', lambda a: hits[a]['strict'] / n_t)
    L += ['', '## L2 — mechanism', '',
          'A1 gains: **%d reach + %d rerank** · losses: %d reach + %d rerank '
          '(twin-credited, @5)' % (gains['reach'], gains['rerank'],
                                   losses['reach'], losses['rerank']), '']
    for dim, title in (('depth', 'stack depth'), ('cue-len', 'cue length'),
                       ('pivot', 'pivot (within-session q1 split)')):
        L.append('| %s | n | A0 | A1 | Δ |' % title)
        L.append('|---|---:|---:|---:|---:|')
        for b in sorted(cell[dim]):
            n, h0, h1 = cell[dim][b]
            L.append('| %s | %d | %.0f%% | %.0f%% | %+.0fpp |'
                     % (b, n, 100 * h0 / n, 100 * h1 / n, 100 * (h1 - h0) / n))
        L.append('')
    L += ['## L3 — stats floor', '',
          '| | A0 | A1 | Δ | A0f | A1a | C1 |', '|---|---:|---:|---:|---:|---:|---:|']
    row('soft_r (pooled)', lambda a: soft_r[a], fmt='%.3f', dfmt='%+.3f', scale=1)
    row('recall p50 ms', lambda a: float(np.percentile(lat[a], 50)),
        fmt='%.0f', dfmt='%+.0f', scale=1)
    row('recall p95 ms', lambda a: float(np.percentile(lat[a], 95)),
        fmt='%.0f', dfmt='%+.0f', scale=1)
    c2 = d.leg_b.get('c2') or {}
    if c2.get('pass') is None:
        # Missing leg_b_report.json, or a zero-sample run: a check that
        # never examined anything renders as NOT RUN, never as ✗.
        c2_txt = 'NOT RUN (%s)' % (
            c2.get('status') or 'no c2 block in leg_b_report.json')
    else:
        c2_txt = '%d/%d mismatches → %s' % (
            len(c2.get('mismatches', [])), c2.get('first_cues', 0),
            '✓' if c2['pass'] else '✗')
    p_pass = delta >= 0.05 and ci[0] > 0
    L += ['',
          '**P-primary (§20.18):** Δ(A1−A0f) = %+.3f, clustered 95%% CI '
          '[%.3f, %.3f] → **%s**' % (delta, ci[0], ci[1],
                                     'PASS' if p_pass else 'FAIL'),
          '**C1 shuffle:** %.3f vs A0f %.3f → %s · **C2 identity:** %s'
          % (soft_r['C1'], soft_r['A0f'],
             '✓' if soft_r['C1'] < soft_r['A0f'] else '✗', c2_txt), '',
          '## Per-item side by side (mean delivered soft@5)', '',
          '| qid | axis | A0 | A1 | Δ | move |', '|---|---|---:|---:|---:|:---:|']
    ups = downs = 0
    for qid in sorted(per_qid):
        a = float(np.mean(per_qid[qid].get(BASE, [np.nan])))
        b = float(np.mean(per_qid[qid].get(CAND, [np.nan])))
        dd = b - a
        ups += dd > 0.01
        downs += dd < -0.01
        L.append('| `%s` | %s | %.3f | %.3f | %+.3f | %s |'
                 % (qid, axis_of.get(qid, '?'), a, b, dd,
                    '↑' if dd > 0.01 else ('↓' if dd < -0.01 else '=')))
    L.append('')
    L.append('Movement: **%d ↑ / %d ↓ / %d =** of %d items.'
             % (ups, downs, len(per_qid) - ups - downs, len(per_qid)))

    out = d.cdir / 'leg_b' / 'SUMMARY.md'
    out.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nreport → %s' % out)


if __name__ == '__main__':
    main()
