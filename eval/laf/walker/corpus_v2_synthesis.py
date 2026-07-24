"""Corpus-v2 synthesis — the four protocol deliverables (node 25cea181 §SYNTHESIS).

Joins corpus_v2_verdicts.jsonl (semantic verdict + stratum per turn) to
corpus_v2_bundles.jsonl (telemetry: static-mix/F0/M_h ranks, strong tier,
gold age). Produces:
  1. per-stratum honest baselines — reach@5 / @25 over VALID golds only,
     split cue (door-1) vs window/session (door-2); echoes EXCLUDED
  2. echo-mislabel rate overall + vs strong tier (protocol pred: mislabels
     concentrate off the strong tier)
  3. failure-mode codebook — bridge-device buckets with counts (VALID misses)
  4. recency cross-tab — verdict × gold-age band → the cohort/episodic/topic
     read on why old golds fail

Read-only. Run: ./dev python3 eval/laf/walker/corpus_v2_synthesis.py
Out: OUT_DIR/corpus_v2_synthesis.md
"""
import json
import re
from collections import Counter, defaultdict

from walker_db import OUT_DIR

V = OUT_DIR / 'corpus_v2_verdicts.jsonl'
B = OUT_DIR / 'corpus_v2_bundles.jsonl'
REPORT = OUT_DIR / 'corpus_v2_synthesis.md'


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    verds = {json.loads(x)['key']: json.loads(x) for x in V.open()}
    bundles = {json.loads(x)['key']: json.loads(x) for x in B.open()}
    # strong tier UNCONDITIONALLY from the index (Turn.strong semantics needs
    # only sel/gold_i) — the bundle telemetry gate is about mix computability
    # and silently dropped 156 rows' strong flag (code-review 2026-07-24)
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    strong_by_key = {'%s/%d/%d' % tuple(t['key']):
                     bool(t['sel'][t['gold_i']]) if t.get('sel') else False
                     for t in idx['turns']}
    rows = []
    for k, v in verds.items():
        b = bundles.get(k)
        if not b:
            continue
        tel = b.get('telemetry') or {}
        g = b.get('gold') or {}
        rows.append({
            'key': k, 'verdict': v['verdict'], 'stratum': v['stratum'],
            'bridge': v.get('bridge'), 'gap': v.get('gap'),
            'style_note': v.get('style_note'), 'rubric': v.get('rubric'),
            'mix': tel.get('mix_rank'), 'f0': tel.get('f0_rank'),
            'mh': tel.get('mh_rank'), 'strong': strong_by_key.get(k, False),
            'age': g.get('age_days'), 'gtype': g.get('type'),
            'v0': b.get('v0_stratum'),
        })
    n = len(rows)
    L = ['# Corpus-v2 synthesis (2026-07-21)', '',
         'Semantic judge pass over the walker gold corpus. n=%d turns '
         '(1 dead-gold dropped). Rubric v3, Sonnet judges, Opus-audited.' % n,
         '']

    # ---- headline ----
    vc = Counter(r['verdict'] for r in rows)
    L += ['## 1. Verdict distribution', '',
          '| verdict | n | share |', '|---|---|---|']
    for k in ('valid', 'echo_mislabel', 'ambiguous'):
        L.append('| %s | %d | %.0f%% |' % (k, vc[k], pct(vc[k], n)))
    L.append('')

    # ---- per-stratum honest baselines (VALID only) ----
    valids = [r for r in rows if r['verdict'] == 'valid']
    L += ['## 2. Per-stratum honest baselines (VALID golds only, echoes excluded)',
          '',
          'reach@k = static λ=0.65 mix rank ≤ k. Door-1 = cue; Door-2 = '
          'window + session (Moments).', '',
          '| stratum | n | reach@5 | reach@25 | median mix-rank |',
          '|---|---|---|---|---|']
    import statistics
    for st in ('cue', 'window', 'session'):
        sub = [r for r in valids if r['stratum'] == st and r['mix'] is not None]
        if not sub:
            continue
        r5 = sum(1 for r in sub if r['mix'] <= 5)
        r25 = sum(1 for r in sub if r['mix'] <= 25)
        med = statistics.median(r['mix'] for r in sub)
        L.append('| %s | %d | %.0f%% | %.0f%% | %.0f |'
                 % (st, len(sub), pct(r5, len(sub)), pct(r25, len(sub)), med))
    # door rollups
    for lbl, sts in (('DOOR-1 (cue)', {'cue'}),
                     ('DOOR-2 (window+session)', {'window', 'session'})):
        sub = [r for r in valids if r['stratum'] in sts and r['mix'] is not None]
        r5 = sum(1 for r in sub if r['mix'] <= 5)
        L.append('| **%s** | %d | **%.0f%%** | %.0f%% | — |'
                 % (lbl, len(sub), pct(r5, len(sub)),
                    pct(sum(1 for r in sub if r['mix'] <= 25), len(sub))))
    L += ['',
          'Contrast: the OLD blended reach@5 counted all %d golds as one '
          'population.' % n,
          '- blended reach@5 over ALL golds: %.0f%%'
          % pct(sum(1 for r in rows if r['mix'] and r['mix'] <= 5),
                sum(1 for r in rows if r['mix'] is not None)),
          '- reach@5 over VALID golds only: %.0f%%'
          % pct(sum(1 for r in valids if r['mix'] and r['mix'] <= 5),
                sum(1 for r in valids if r['mix'] is not None)),
          '']

    # ---- echo rate vs strong tier ----
    L += ['## 3. Echo-mislabel rate vs strong tier', '',
          'strong = soft-gold AND Haiku-picked that turn. Protocol '
          'prediction: mislabels concentrate OFF the strong tier.', '',
          '| tier | n | echo% | valid% |', '|---|---|---|---|']
    for lbl, pred in (('strong', lambda r: r['strong']),
                      ('non-strong', lambda r: not r['strong'])):
        sub = [r for r in rows if pred(r)]
        e = sum(1 for r in sub if r['verdict'] == 'echo_mislabel')
        va = sum(1 for r in sub if r['verdict'] == 'valid')
        L.append('| %s | %d | %.0f%% | %.0f%% |'
                 % (lbl, len(sub), pct(e, len(sub)), pct(va, len(sub))))
    L.append('')

    # ---- codebook: bridge-device buckets over VALID misses ----
    valid_miss = [r for r in valids if r['mix'] and r['mix'] > 5]
    buckets = {
        'graph-walk': r'graph|walk|neighbor|1-hop|traver',
        'situation-lane / re-enrich': r'situation|re-enrich|reenrich|re-write|rewrite',
        'lexical / idf': r'lexical|idf|rare token|keyword',
        'episodic recency / same-session': r'episodic|recency|same-session|recent',
        'conversation-window (M_h)': r'window|m_h|mh|conversation-window|prior turn',
        'style-recall (Tom-pattern)': r'style|pattern node|tom-pattern|greenlight|handoff|discipline',
        'node-class prior': r'node-class|node class|demote|quote|reflection prior',
        'running session field': r'running|session field|persistent field',
        'query segmentation': r'segment|per-question|per-clause|split',
    }
    codebook = Counter()
    for r in valid_miss:
        txt = (r['bridge'] or '').lower()
        hit = False
        for name, pat in buckets.items():
            if re.search(pat, txt):
                codebook[name] += 1
                hit = True
        if not hit:
            codebook['(other / unbucketed)'] += 1
    L += ['## 4. Failure-mode codebook — bridge devices on VALID misses (n=%d)'
          % len(valid_miss), '',
          'Which held/missing device the judge named as the bridge. Multi-'
          'label (a bridge can name several).', '',
          '| device | count | share of valid-misses |', '|---|---|---|']
    for name, c in codebook.most_common():
        L.append('| %s | %d | %.0f%% |' % (name, c, pct(c, len(valid_miss))))
    L.append('')

    # ---- recency cross-tab ----
    bands = [('≤1d', 0, 1), ('1-7d', 1, 7), ('7-21d', 7, 21),
             ('21-45d', 21, 45), ('>45d', 45, 1e9)]
    L += ['## 5. Recency cross-tab — verdict & valid-reach by gold age', '',
          '| age band | n | echo% | valid% | valid reach@5 |',
          '|---|---|---|---|---|']
    for lbl, lo, hi in bands:
        sub = [r for r in rows if r['age'] is not None and lo <= r['age'] < hi]
        if not sub:
            continue
        e = sum(1 for r in sub if r['verdict'] == 'echo_mislabel')
        va = [r for r in sub if r['verdict'] == 'valid']
        vr = [r for r in va if r['mix'] is not None]
        vr5 = sum(1 for r in vr if r['mix'] <= 5)
        L.append('| %s | %d | %.0f%% | %.0f%% | %.0f%% |'
                 % (lbl, len(sub), pct(e, len(sub)), pct(len(va), len(sub)),
                    pct(vr5, len(vr))))
    L.append('')

    # ---- v0 (mechanical) vs semantic cross-tab ----
    L += ['## 6. Mechanical v0 stratum vs semantic verdict', '',
          'Where the old strata_v0 mechanical bins land under semantic '
          'judgment.', '',
          '| v0 stratum | n | echo% | valid% | of valids: cue/win/sess |',
          '|---|---|---|---|---|']
    for v0 in ('CUE-SUFF', 'MOMENT-DEP', 'NEITHER'):
        sub = [r for r in rows if r['v0'] == v0]
        if not sub:
            continue
        e = sum(1 for r in sub if r['verdict'] == 'echo_mislabel')
        va = [r for r in sub if r['verdict'] == 'valid']
        sc = Counter(r['stratum'] for r in va)
        L.append('| %s | %d | %.0f%% | %.0f%% | %d/%d/%d |'
                 % (v0, len(sub), pct(e, len(sub)), pct(len(va), len(sub)),
                    sc['cue'], sc['window'], sc['session']))
    L.append('')

    # provenance caveat
    rc = Counter(r['rubric'] for r in rows)
    L += ['## Provenance', '',
          '- rubric mix: %s' % dict(rc),
          '- The %d v2-rubric rows are front-half VALIDS (kept un-re-judged: '
          'loosening only moves echo→valid, never reverse). Their VERDICTS '
          'are safe; their STRATA were assigned pre-v3 (anaphora rule was '
          'already present in v2, so drift is second-order). Re-judging them '
          'for stratum-perfect consistency = ~3.6M more Sonnet tokens if '
          'wanted.' % rc.get('v2', 0), '']

    L += ['## Verdict — what the corpus says', '',
          '**Echo-mislabel is the dominant defect: 51% of walker "gold" is '
          'response-echo, not helpful recall.** The soft-labeler minted a '
          'label whenever a node shared vocabulary with Anchor\'s NEXT '
          'response; half the time that node did not serve the moment.', '',
          '**Honest reach is 50%, not 31%.** The blended 31% was dragged '
          '~19pp by echo-mislabels no recall system should surface. Over '
          'genuine golds, static-mix reach@5 is 50%.', '',
          '**The strong tier is the trustworthy sub-corpus** (22% echo / 75% '
          'valid vs 62% / 37% off-tier) — soft ∩ Haiku-picked is a ~75%-clean '
          'label, exactly the protocol prediction. Build and measure against '
          'the strong tier, not the blend.', '',
          '**Door-2 is real and under-served.** Valid golds split cue 388 / '
          'window 371 / session 153. Session-stratum Moments reach only 33% '
          '(median rank 14) vs 50-57% for cue/window — the genuine Moment '
          'golds are exactly what the static mix cannot reach. That is the '
          'running-field case, now quantified on clean golds.', '',
          '**Recency verdict: cohort-drift + episodic-recency, NOT topic-'
          'drift.** Two separable effects: (a) echo-rate CLIMBS with age '
          '(26%→63%) — old nodes accreted content that spuriously matches '
          'random future responses, so old golds are disproportionately '
          'mislabels (cohort/labeling artifact, not recall failure); (b) '
          'among VALID golds, reach@5 STEPS DOWN 70%(≤1d)→60%(1-7d)→~40%'
          '(>7d) then PLATEAUS — the fresh-gold advantage is episodic-lane '
          'presence (same-session recency), and once it decays reach settles '
          'at the semantic/lexical floor. The plateau past 21d (not '
          'monotonic decay) rules out topic-drift as the driver — it is a '
          'binary episodic-lane on/off, not gradual topical distance.', '',
          '**Mechanical v0 was mostly wrong about NEITHER.** The v0 NEITHER '
          'bin (0% reach) is 73% echo-mislabel — its "0% reach" was largely '
          'an INVALID exam (non-golds), not mechanism failure, confirming '
          '7e8f82e3. CUE-SUFF is 60% valid; the honest door-1 exam lives '
          'there minus its echoes.', '',
          '**Top recoverable levers (bridge codebook on valid misses):** '
          'graph-walk (44% — the inert graph operator, 54777ca7) and '
          'situation-lane/re-enrichment (40%) dominate; episodic recency '
          '(27%) third. Style-recall (11%) is the door-2/terse-cue lever.',
          '']

    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('\nwrote %s' % REPORT)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
