"""Leg A arm scoring on an EXTERNAL pooled corpus (§20.18 P-primary/P-secondary).

Applies the FROZEN definitive-fit weights (definitive_fit.json, walker v7 —
never refit here) to the pooled walker's (cue, candidate) rows and reports:

  P-primary   pooled soft_r per arm (Pearson vs soft_max, the fit instrument
              verbatim) + session-clustered bootstrap 95% CI on r(A1)−r(A0f).
              PASS bar (pre-committed): A1 ≥ A0f + 0.05, CI excluding 0.
  P-secondary rank-within-pool at the evidence op-cues: the candidate pool is
              the build's production top-25, so Leg A measures RERANKING of
              that pool; true reach differences are Leg B's (engine path).
              Exact counts, no percentage theater.

Arms (feature space of the fit — 5 lanes × 17 slots + M_e_f):
  A0   production actual: the build's own pool_score (candidates table)
  A0f  S_content restricted to op·j0 slots (+M_e_f — not a slot; kept in
       every fitted arm so A1−A0f isolates the HISTORY slots)
  A1   S_content full table (THE hypothesis)
  A1t  A1 with slot |w| < 0.10 zeroed (pre-registered trim check)
  A1s  S_full (pick/enc retained — Leg-A-only arm)

Run (env-glue like every pooled walker phase):
  WALKER_OUT_DIR=<corpus>/walker ./dev python3 eval/laf/walker/external_score.py \
      --manifest ~/AgentsContext/eval-corpus/<hash>/manifest.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, WALKER_DIR, open_walker
from q1_sweep import load, gate_provenance
from definitive_fit import FEATURES, turn_features

FIT = WALKER_DIR / 'definitive_fit.json'
TRIM_ABS = 0.10
BOOT_N = 2000
BOOT_SEED = 42
REPORT = OUT_DIR / 'external_score.md'
OUT = OUT_DIR / 'external_score.json'

_DATE_PREFIX = re.compile(r'^\[Current date:[^\]]*\]\s*')


def weight_vector(wdict, j0_only=False, trim=0.0):
    """Map a fit weight dict onto the FEATURES basis. M_e_f is not a slot:
    it survives j0_only and is never trimmed."""
    v = np.zeros(len(FEATURES))
    for i, name in enumerate(FEATURES):
        w = wdict.get(name, 0.0)
        if name != 'M_e_f':
            if j0_only and not name.endswith('op0'):
                w = 0.0
            if trim and abs(w) < trim:
                w = 0.0
        v[i] = w
    return v


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    return float(np.corrcoef(x[m], y[m])[0, 1])


def evidence_cues(manifest_path):
    """(sid, stripped content, gold node ids) per has_answer USER turn of the
    ANSWERABLE items — the sid scheme is build_corpus._pooled_session_plan's."""
    manifest = json.loads(Path(manifest_path).read_text())
    gold_by_qid = {it['qid']: [m['node_id'] for m in it['gold_scan']['matches']]
                   for it in manifest['items'] if it['answerable']}
    oracle = json.loads((Path(__file__).resolve().parents[2] / 'longmem' /
                         'data' / manifest['config']['oracle']).read_text())
    out = []
    for item in oracle:
        qid = item['question_id']
        if qid not in gold_by_qid:
            continue
        for sess_idx, session in enumerate(item.get('haystack_sessions', [])):
            h = hashlib.sha1(('%s|%d' % (qid, sess_idx)).encode()).hexdigest()
            sid = 'i%s-%s-s%d' % (h[:7], qid, sess_idx)
            for t in session:
                if t.get('role') == 'user' and t.get('has_answer'):
                    out.append((qid, sid, t['content'], gold_by_qid[qid]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    args = p.parse_args()

    walker = open_walker()
    gate_provenance(walker)
    fit = json.loads(FIT.read_text())
    turns = {td.key: td for td in load(walker)}

    arms = {
        'A0f': weight_vector(fit['weights']['S_content'], j0_only=True),
        'A1':  weight_vector(fit['weights']['S_content']),
        'A1t': weight_vector(fit['weights']['S_content'], trim=TRIM_ABS),
        'A1s': weight_vector(fit['weights']['S_full']),
    }

    pool_score = {}
    for row in walker.execute(
            'SELECT session_id, epoch, seq, node_id, pool_score '
            'FROM candidates WHERE node_id IS NOT NULL'):
        pool_score[(row[0], row[1], row[2], row[3])] = row[4]

    # ── per-turn scores, pooled rows ────────────────────────────────────
    per_sess = {}       # sid → {arm: (xs, ys)} for clustered bootstrap
    scores_by_key = {}  # turn key → {arm: np.array} for P-secondary
    for key, td in turns.items():
        X = turn_features(td)
        row_scores = {a: X @ w for a, w in arms.items()}
        row_scores['A0'] = np.array(
            [pool_score.get((*key, nid), np.nan) or np.nan
             for nid in td.cands], dtype=float)
        scores_by_key[key] = row_scores
        sess = key[0]
        bucket = per_sess.setdefault(sess, {a: ([], []) for a in row_scores})
        for a, s in row_scores.items():
            m = np.isfinite(td.soft) & np.isfinite(s)
            if m.any():
                bucket[a][0].append(s[m])
                bucket[a][1].append(td.soft[m])

    arm_names = ['A0', 'A0f', 'A1', 'A1t', 'A1s']
    pooled = {a: (np.concatenate([x for s in per_sess.values()
                                  for x in s[a][0]] or [np.array([])]),
                  np.concatenate([y for s in per_sess.values()
                                  for y in s[a][1]] or [np.array([])]))
              for a in arm_names}
    soft_r = {a: pearson(*pooled[a]) for a in arm_names}

    # session-clustered bootstrap on r(A1) − r(A0f)
    rng = np.random.default_rng(BOOT_SEED)
    sessions = sorted(per_sess)
    diffs = []
    for _ in range(BOOT_N):
        pick = rng.choice(len(sessions), len(sessions), replace=True)
        xs1, ys1, xs0, ys0 = [], [], [], []
        for i in pick:
            s = per_sess[sessions[i]]
            xs1 += s['A1'][0]; ys1 += s['A1'][1]
            xs0 += s['A0f'][0]; ys0 += s['A0f'][1]
        diffs.append(pearson(np.concatenate(xs1), np.concatenate(ys1)) -
                     pearson(np.concatenate(xs0), np.concatenate(ys0)))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    delta = soft_r['A1'] - soft_r['A0f']
    p_primary_pass = bool(delta >= 0.05 and lo > 0)

    # ── P-secondary: rank-within-pool at evidence op-cues ───────────────
    ops_by_sid = {}
    for sid, epoch, seq, op in walker.execute(
            'SELECT session_id, epoch, seq, op_text FROM turns WHERE labeled=1'):
        ops_by_sid.setdefault(sid, []).append(
            ((sid, epoch, seq), _DATE_PREFIX.sub('', op or '')))
    cues = evidence_cues(args.manifest)
    sec_rows = []
    for qid, sid, content, gold_ids in cues:
        key = next((k for k, op in ops_by_sid.get(sid, [])
                    if op.startswith(content[:200])), None)
        if key is None or key not in scores_by_key:
            sec_rows.append({'qid': qid, 'sid': sid, 'status': 'cue_not_scored'})
            continue
        td = turns[key]
        in_pool = [i for i, nid in enumerate(td.cands) if nid in set(gold_ids)]
        if not in_pool:
            sec_rows.append({'qid': qid, 'sid': sid, 'status': 'gold_not_in_pool'})
            continue
        entry = {'qid': qid, 'sid': sid, 'status': 'ok', 'ranks': {}}
        for a in arm_names:
            s = scores_by_key[key][a]
            order = np.argsort(-np.where(np.isfinite(s), s, -np.inf))
            rank = min(int(np.where(order == i)[0][0]) + 1 for i in in_pool)
            entry['ranks'][a] = rank
        sec_rows.append(entry)

    ok = [r for r in sec_rows if r['status'] == 'ok']
    sec_counts = {a: {'r@5': sum(1 for r in ok if r['ranks'][a] <= 5),
                      'r@25': sum(1 for r in ok if r['ranks'][a] <= 25)}
                  for a in arm_names}

    # ── report ──────────────────────────────────────────────────────────
    n_rows = int(np.isfinite(pooled['A1'][0]).sum())
    lines = ['# external_score — Leg A arm scoring (frozen fit, %d turns, '
             '%d pooled rows)' % (len(turns), n_rows), '',
             '| arm | soft_r |', '|---|---|']
    lines += ['| %s | %.4f |' % (a, soft_r[a]) for a in arm_names]
    lines += ['',
              'Δ(A1 − A0f) = %.4f, clustered bootstrap 95%% CI [%.4f, %.4f] '
              '(%d sessions, %d resamples)' % (delta, lo, hi,
                                               len(sessions), BOOT_N),
              '',
              '**P-primary (A1 ≥ A0f + 0.05, CI > 0): %s**'
              % ('PASS' if p_primary_pass else 'FAIL'),
              '',
              '## P-secondary — rank-within-pool at evidence op-cues '
              '(pool = build top-25; true reach is Leg B)',
              '',
              'cues: %d total, %d scored, %d cue_not_scored, '
              '%d gold_not_in_pool' % (
                  len(sec_rows), len(ok),
                  sum(1 for r in sec_rows if r['status'] == 'cue_not_scored'),
                  sum(1 for r in sec_rows if r['status'] == 'gold_not_in_pool')),
              '', '| arm | rank≤5 | rank≤25 |', '|---|---|---|']
    lines += ['| %s | %d/%d | %d/%d |' % (a, sec_counts[a]['r@5'], len(ok),
                                          sec_counts[a]['r@25'], len(ok))
              for a in arm_names]
    REPORT.write_text('\n'.join(lines) + '\n')
    OUT.write_text(json.dumps({
        'soft_r': soft_r, 'delta_a1_a0f': delta, 'ci95': [float(lo), float(hi)],
        'p_primary_pass': p_primary_pass, 'n_rows': n_rows,
        'n_sessions': len(sessions), 'secondary': {'counts': sec_counts,
                                                   'rows': sec_rows},
    }, indent=1))
    print('\n'.join(lines))
    print('\nreport → %s' % REPORT)


if __name__ == '__main__':
    main()
