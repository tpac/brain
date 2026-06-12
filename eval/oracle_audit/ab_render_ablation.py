#!/usr/bin/env python3
"""AREA-2 ABLATION: does the lean selection render change Haiku's picks?

Real Haiku selection calls (single-shot, no tools — isolates the RENDER effect;
the agentic loop would confound with tool-fetched candidates). Same 25 enriched
candidates per query, four render arms:
  full      — production HAIKU_FORMAT (control)
  lean      — HAIKU_FORMAT_LEAN (ships behind BRAIN_HAIKU_RENDER=lean)
  lean_noedge   — lean with edge_limit=0 (does edge signal matter for selection?)
  lean_c150     — lean with content_limit=150 (what is the 300-char content worth?)

Metrics per arm: selection agreement vs control (Jaccard + exact-set rate),
gold_essential coverage of the SELECTED set, prompt tokens, Haiku latency.
Frame omitted in all arms (constant across arms — isolates candidate render).
Uses the production-ACTIVE surface system prompt from the interactions table.

Usage: ./dev python3 eval/oracle_audit/ab_render_ablation.py"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']

_ALL_ARMS = {
    'full':        {},      # sentinel: use HAIKU_FORMAT
    'full2':       {},      # identical to full — run both for the noise floor
    'lean':        {'lean': True},
    'lean_noedge': {'lean': True, 'edge_limit': 0},
    'lean_c150':   {'lean': True, 'content_limit': 150},
}
# BRAIN_ABLATION_ARMS="full,full2" selects a subset (default: original 4)
_sel = os.environ.get('BRAIN_ABLATION_ARMS', 'full,lean,lean_noedge,lean_c150')
ARMS = {a: _ALL_ARMS[a] for a in _sel.split(',') if a in _ALL_ARMS}


def toks(s):
    return len(s) / 3.5


def render_block(cands, arm_cfg):
    from servers.contract import render_rich_node
    from servers.scales.s1.surface_contract import HAIKU_FORMAT, HAIKU_FORMAT_LEAN
    base = dict(HAIKU_FORMAT_LEAN) if arm_cfg.get('lean') else dict(HAIKU_FORMAT)
    for k, v in arm_cfg.items():
        if k != 'lean':
            base[k] = v
    blocks = []
    for i, c in enumerate(cands, 1):
        header = "#%d" % i
        score = c.get('score', 0)
        if score:
            header += " (match:%.2f)" % min(score, 1.0)
        blocks.append(header + "\n" + render_rich_node(c, base))
    return "\n\n".join(blocks)


with IsolatedBrain() as env:
    b = env.brain
    import anthropic
    from servers.scales.s1.surface_contract import (
        SURFACE_SELECTION_SCHEMA, SURFACE_MODEL)
    client = anthropic.Anthropic()
    sys_prompt = b.get_interaction_prompt('surface')
    assert sys_prompt, "surface interaction prompt missing"

    # Stage 1: candidates per query (once — shared across arms)
    per_q = []
    for q in QS:
        ess = {g[:8] for g in q.get('gold_essential', [])}
        if not ess:
            continue
        out = b.recall(query=q['query'], limit=25)
        raw = out.get('results', []) if isinstance(out, dict) else (out or [])
        ids = [r.get('id', '') for r in raw]
        rich = b.get_node(ids)
        rich_map = rich if isinstance(rich, dict) else {}
        cands = []
        for r in raw:
            nd = rich_map.get(r.get('id', '')) or dict(r)
            if 'id' not in nd:
                nd['id'] = r.get('id', '')
            nd['score'] = r.get('effective_activation', 0)
            cands.append(nd)
        per_q.append({'q': q, 'ess': ess, 'cands': cands})

    def select(qrec, arm):
        block = render_block(qrec['cands'], ARMS[arm])
        user = ("## Operator message\n%s\n\n## Candidate memories\n%s\n\n"
                "Select the 3-5 most relevant candidates."
                % (qrec['q']['query'], block))
        t0 = time.time()
        try:
            resp = client.messages.create(
                model=SURFACE_MODEL, max_tokens=1000,
                system=sys_prompt,
                messages=[{"role": "user", "content": user}],
                output_config={'format': {'type': 'json_schema',
                                          'schema': SURFACE_SELECTION_SCHEMA}})
            raw = resp.content[0].text
            sel = {s.get('id', '')[:8] for s in json.loads(raw).get('selected', [])}
        except Exception as e:
            return {'error': str(e)[:120], 'sel': set(), 'ms': (time.time()-t0)*1000,
                    'tok': toks(user)}
        return {'sel': sel, 'ms': (time.time()-t0)*1000, 'tok': toks(user), 'error': None}

    # Stage 2: all (query, arm) selections in parallel
    jobs = [(qi, arm) for qi in range(len(per_q)) for arm in ARMS]
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(select, per_q[qi], arm): (qi, arm) for qi, arm in jobs}
        for fut in futs:
            qi, arm = futs[fut]
            results[(qi, arm)] = fut.result()

    # Stage 3: score
    print("\n=== render ablation — %d queries × %d arms (single-shot Haiku) ===" % (len(per_q), len(ARMS)))
    print("%-12s %9s %9s %10s %10s %9s %7s" % (
        "arm", "agree-J", "same-set", "gold-hit", "gold-cov", "tok/call", "ms/call"))
    for arm in ARMS:
        js, same, gh, gc_n, gc_d, tks, ms, errs = [], 0, 0, 0, 0, [], [], 0
        for qi, qrec in enumerate(per_q):
            r = results[(qi, arm)]
            c = results[(qi, 'full')]
            if r['error']:
                errs += 1
                continue
            if not c['error'] and arm != 'full':
                inter = len(r['sel'] & c['sel'])
                union = len(r['sel'] | c['sel']) or 1
                js.append(inter / union)
                same += 1 if r['sel'] == c['sel'] else 0
            ess = qrec['ess']
            hit = len(r['sel'] & ess)
            gh += 1 if hit > 0 else 0
            gc_n += hit
            gc_d += len(ess)
            tks.append(r['tok']); ms.append(r['ms'])
        n = len(per_q) - errs
        print("%-12s %9s %9s %10s %10s %9.0f %7.0f%s" % (
            arm,
            ("%.2f" % (sum(js)/len(js))) if js else "—",
            ("%d/%d" % (same, n)) if arm != 'full' else "—",
            "%d/%d" % (gh, n),
            "%d/%d (%.0f%%)" % (gc_n, gc_d, 100.0*gc_n/max(gc_d, 1)),
            sum(tks)/max(len(tks), 1), sum(ms)/max(len(ms), 1),
            ("  errors:%d" % errs) if errs else ""))

    print("\n--- divergence detail: queries where an arm's gold-hit differs from full ---")
    for arm in ARMS:
        if arm == 'full':
            continue
        for qi, qrec in enumerate(per_q):
            r, c = results[(qi, arm)], results[(qi, 'full')]
            if r['error'] or c['error']:
                continue
            rh, ch = len(r['sel'] & qrec['ess']), len(c['sel'] & qrec['ess'])
            if rh != ch:
                mark = "BETTER" if rh > ch else "WORSE"
                print("  [%-11s] #%-4s %-8s gold-in-selected %d→%d  %s" % (
                    arm, qrec['q']['id'], qrec['q']['mode'], ch, rh, mark))
