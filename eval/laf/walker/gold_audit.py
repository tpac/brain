"""GOLD-VALIDITY AUDIT (Tom's gate, 2026-07-21): before any 'unlearning'
re-runs, double-verify the corpus and golds — and put REAL examples in
front of the operator so cosine-soft-vs-Haiku-picks becomes a judgment
call on evidence, not vibes.

The gold rule (verbatim from field_cache_build): per labeled turn, the
top-soft candidate, admitted only if its soft is >= the corpus-wide 90th
percentile. soft = max cosine(candidate's 6 content views, Anchor's actual
next response) — soft_usage.py.

Sections:
  1. composition — gold node type / age-at-turn / Haiku-outcome / fetched_by
  2. soft-vs-picks agreement — did Haiku itself pick the gold? what did it
     pick instead, and how do those picks score on soft?
  3. examples (OUT_DIR/gold_audit.md) — stratified: agree / disagree /
     near-threshold, each with the turn text, the response snippet, the
     gold, and Haiku's picks — the eyeball material.

Run:    ./dev python3 eval/laf/walker/gold_audit.py
Pool60: BRAIN_DB_DIR=... WALKER_OUT_DIR=... (same as every walker run)
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from walker_db import OUT_DIR, open_walker, open_brain_ro

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from q1_sweep import load, gate_provenance                           # noqa: E402
from servers.recall_laf import _unit                                 # noqa: E402

REPORT = OUT_DIR / 'gold_audit.md'
N_AGREE, N_DISAGREE, N_NEAR = 10, 15, 5
SNIP = 220


def iso_days(a, b):
    try:
        return (datetime.fromisoformat(a.replace('Z', '+00:00'))
                - datetime.fromisoformat(b.replace('Z', '+00:00'))).days
    except Exception:
        return None


def main():
    walker = open_walker()
    gate_provenance(walker)
    turns = load(walker)

    # q_vec gate included: field_cache_build admits a gold only when the
    # turn's q_vec unit-normalizes — same population, or the audit's
    # percentages cover turns every probe excludes (review 2026-07-21)
    tmeta = dict((((s, e, q), (op or '', an or '', ts)) for s, e, q, ts,
                  op, an, qv in walker.execute(
                      "SELECT session_id, epoch, seq, ts, op_text, "
                      "anchor_text, q_vec FROM turns WHERE labeled=1")
                  if _unit(qv) is not None))
    crow = {}
    for s, e, q, nid, out, fb, cat in walker.execute(
            "SELECT session_id, epoch, seq, node_id, outcome, fetched_by, "
            "node_created_at FROM candidates WHERE node_id IS NOT NULL"):
        crow[(s, e, q, nid)] = (out, fb, cat)
    walker.close()

    # gold derivation — verbatim the field_cache_build rule
    allsoft = np.concatenate([td.soft[np.isfinite(td.soft)]
                              for td in turns if np.isfinite(td.soft).any()])
    hi = float(np.percentile(allsoft, 90))
    golds = []          # (td, gold_i)
    for td in turns:
        if not np.isfinite(td.soft).any():
            continue
        g = int(np.nanargmax(td.soft))
        if td.soft[g] >= hi and td.key in tmeta:
            golds.append((td, g))
    print('labeled turns %d · gold turns %d · hi(90th pct) = %.3f'
          % (len(turns), len(golds), hi))

    # node metadata from the brain (read-only)
    need = set()
    for td, g in golds:
        need.update(td.cands)
    bro = open_brain_ro()
    meta = {}
    for chunk in (list(need)[i:i + 500] for i in range(0, len(need), 500)):
        for nid, title, typ in bro.execute(
                'SELECT id, title, type FROM nodes WHERE id IN (%s)'
                % ','.join('?' * len(chunk)), chunk):
            meta[nid] = (title, typ)
    bro.close()

    # ── 1+2. composition + agreement ────────────────────────────────────
    types, ages, outc, fbs = Counter(), [], Counter(), Counter()
    agree, disagree, near = [], [], []
    for td, g in golds:
        gid = td.cands[g]
        title, typ = meta.get(gid, ('?', '?'))
        out, fb, cat = crow.get((*td.key, gid), (None, None, None))
        ts = tmeta[td.key][2]
        types[typ] += 1
        outc[out or '?'] += 1
        fbs[fb or 'pool'] += 1
        if cat:
            d = iso_days(ts, cat)
            if d is not None:
                ages.append(d)
        rec = (td, g, gid, out)
        (agree if out == 'selected' else disagree).append(rec)
        if td.soft[g] < hi * 1.05:
            near.append(rec)

    n = len(golds)
    print('\n== 1. gold composition ==')
    print('  type: ' + ' · '.join('%s %d (%.0f%%)' % (t, c, 100 * c / n)
                                  for t, c in types.most_common(8)))
    a = np.array(ages)
    print('  age at turn (days): median %.1f · <1d %.0f%% · >7d %.0f%% '
          '(n=%d)' % (np.median(a), 100 * (a < 1).mean(),
                      100 * (a > 7).mean(), len(a)))
    print('  gold Haiku-outcome: ' + ' · '.join(
        '%s %d (%.0f%%)' % (k, c, 100 * c / n) for k, c in outc.items()))
    print('  gold fetched_by: ' + ' · '.join(
        '%s %d' % (k, c) for k, c in fbs.most_common()))
    print('\n== 2. soft-gold vs Haiku picks ==')
    print('  Haiku picked the gold: %d/%d (%.0f%%) · dropped it: %d '
          '(%.0f%%)' % (len(agree), n, 100 * len(agree) / n,
                        len(disagree), 100 * len(disagree) / n))

    # ── 3. examples ─────────────────────────────────────────────────────
    def ex_block(td, g, gid, out):
        op, an, ts = tmeta[td.key]
        title, typ = meta.get(gid, ('?', '?'))
        _, fb, cat = crow.get((*td.key, gid), (None, None, None))
        age = iso_days(ts, cat) if cat else None
        lines = ['### %s · soft %.3f · %s' % ('/'.join(map(str, td.key)),
                                              td.soft[g], ts[:16]),
                 '**Tom:** %s' % op[:SNIP].replace('\n', ' '),
                 '**Anchor (next):** %s' % an[:SNIP].replace('\n', ' '),
                 '**GOLD [%s]** %s — age %sd — Haiku %s'
                 % (typ, title, age, out or '?')]
        picks = [(i, nid) for i, nid in enumerate(td.cands)
                 if crow.get((*td.key, nid), (None,))[0] == 'selected']
        if picks:
            lines.append('**Haiku picked:**')
            for i, nid in picks[:6]:
                t2, ty2 = meta.get(nid, ('?', '?'))
                sv = td.soft[i]
                lines.append('  - [%s] %s — soft %s'
                             % (ty2, t2, ('%.3f' % sv)
                                if np.isfinite(sv) else 'n/a'))
        else:
            lines.append('**Haiku picked:** (nothing this turn)')
        return '\n'.join(lines)

    # deterministic spread: sort by soft desc, take evenly spaced
    def spread(rows, k):
        rows = sorted(rows, key=lambda r: -r[0].soft[r[1]])
        if len(rows) <= k:
            return rows
        idx = np.linspace(0, len(rows) - 1, k).astype(int)
        return [rows[i] for i in idx]

    out = ['# Gold-validity audit — %s\n' % OUT_DIR,
           'gold rule: top-soft per turn, admitted at soft >= %.3f '
           '(90th pct). soft = max cosine(node content views, actual '
           'next response).\n' % hi,
           '\n## A. Haiku PICKED the gold (agree) — %d/%d\n' % (len(agree),
                                                                n)]
    out += [ex_block(*r) for r in spread(agree, N_AGREE)]
    out += ['\n## B. Haiku DROPPED the gold (disagree) — %d/%d\n'
            % (len(disagree), n)]
    out += [ex_block(*r) for r in spread(disagree, N_DISAGREE)]
    out += ['\n## C. Near-threshold golds (soft < 1.05×hi) — %d total\n'
            % len(near)]
    out += [ex_block(*r) for r in spread(near, N_NEAR)]
    REPORT.write_text('\n\n'.join(out))
    print('\nwrote %s (%d examples)' % (
        REPORT, len(spread(agree, N_AGREE)) + len(spread(disagree,
                                                         N_DISAGREE))
        + len(spread(near, N_NEAR))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
