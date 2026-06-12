#!/usr/bin/env python3
"""PROMPT ABLATION v10 vs v11 — drives the REAL agentic surface loop
(_call_surface_agentic: tools fire against the isolated brain, the topical
admission floor runs) per arm. Measures what v11 claims to change:
tool fire rate (trigger taxonomy), gold coverage (must not regress),
latency, floor drops. Render = production default (lean) for both arms.

Arms = surface interaction versions read from the isolated brain's own
interactions table (v11 must be registered before running).
Usage: ./dev python3 eval/oracle_audit/ab_prompt_ablation.py"""
import os, sys, json, time, copy
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = [q for q in json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
      if q.get('gold_essential')]
ARM_VERSIONS = (10, 11)


with IsolatedBrain() as env:
    b = env.brain
    import anthropic
    from servers.scales.s1.surface import _call_surface_agentic, _parse_surfacer_json
    from servers.scales.s1.surface_contract import SURFACE_MODEL, build_surface_prompt
    client = anthropic.Anthropic()

    prompts = {}
    for v in ARM_VERSIONS:
        row = b.logs_conn.execute(
            "SELECT template FROM interactions WHERE name='surface' AND version=?",
            (v,)).fetchone()
        assert row, "surface v%d not registered" % v
        prompts[v] = row[0]

    # Stage 1: enriched candidates per query (shared across arms)
    per_q = []
    for q in QS:
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
        per_q.append({'q': q, 'ess': {g[:8] for g in q['gold_essential']}, 'cands': cands})

    def run_one(qrec, version):
        cands = copy.deepcopy(qrec['cands'])
        user_content, max_tokens = build_surface_prompt(cands, qrec['q']['query'])
        t0 = time.time()
        try:
            raw, tool_trace = _call_surface_agentic(
                client, b, cands, prompts[version], user_content,
                max_tokens, 'ablation-%d' % version, SURFACE_MODEL)
        except Exception as e:
            return {'error': str(e)[:120]}
        parsed = _parse_surfacer_json(raw) or {}
        sel = {(s.get('id') or '')[:8] for s in parsed.get('selected', [])}
        calls = [c for rnd in tool_trace for c in (rnd.get('tool_calls') or [])]
        return {'sel': sel, 'ms': (time.time() - t0) * 1000,
                'calls': calls, 'fired': bool(calls),
                'floor_dropped': sum(c.get('dropped_below_floor', 0) for c in calls),
                'error': None}

    jobs = [(qi, v) for qi in range(len(per_q)) for v in ARM_VERSIONS]
    results = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_one, per_q[qi], v): (qi, v) for qi, v in jobs}
        for fut in futs:
            results[futs[fut]] = fut.result()

    print("\n=== PROMPT ABLATION — real agentic loop, %d queries × v10/v11 ===" % len(per_q))
    print("%-5s %8s %10s %9s %10s %11s %9s" % (
        "arm", "fired", "calls/rec", "gold-hit", "gold-cov", "floor-drop", "ms/call"))
    for v in ARM_VERSIONS:
        fired = tot_calls = gh = gcn = gcd = fd = errs = 0
        ms = []
        for qi, qrec in enumerate(per_q):
            r = results[(qi, v)]
            if r.get('error'):
                errs += 1
                continue
            fired += 1 if r['fired'] else 0
            tot_calls += len(r['calls'])
            hit = len(r['sel'] & qrec['ess'])
            gh += 1 if hit > 0 else 0
            gcn += hit
            gcd += len(qrec['ess'])
            fd += r['floor_dropped']
            ms.append(r['ms'])
        n = len(per_q) - errs
        print("v%-4d %7s %10.1f %9s %10s %11d %9.0f%s" % (
            v, "%d/%d" % (fired, n), tot_calls / max(n, 1),
            "%d/%d" % (gh, n),
            "%d/%d (%.0f%%)" % (gcn, gcd, 100.0 * gcn / max(gcd, 1)),
            fd, sum(ms) / max(len(ms), 1),
            ("  errors:%d" % errs) if errs else ""))

    print("\n--- per-query gold deltas v10→v11 ---")
    any_d = False
    for qi, qrec in enumerate(per_q):
        a, c = results[(qi, 11)], results[(qi, 10)]
        if a.get('error') or c.get('error'):
            continue
        ah, ch = len(a['sel'] & qrec['ess']), len(c['sel'] & qrec['ess'])
        if ah != ch:
            any_d = True
            print("  #%-4s %-8s gold-in-selected %d→%d  %s" % (
                qrec['q']['id'], qrec['q']['mode'], ch, ah,
                "BETTER" if ah > ch else "WORSE"))
    if not any_d:
        print("  (no gold differences)")
