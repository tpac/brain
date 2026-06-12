#!/usr/bin/env python3
"""AREA-2 RECON (read-only): where do the surface prompt's tokens actually go?

For a sample of control-corpus queries: run recall (limit 25), render every
candidate exactly as Haiku sees it (format_candidate_for_surface / HAIKU_FORMAT),
classify each rendered line into components, and report the token distribution —
per candidate block and for the whole assembled prompt (frame + conversation +
candidates). Token estimate: chars/3.5 (distribution shares are what matter).

No behavior change, no writes. Usage: ./dev python3 eval/oracle_audit/ab_render_recon.py"""
import os, sys, json, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain   # noqa: E402

QS = [q for q in json.load(open(f'{ROOT}/eval/oracle_audit/control_corpus.json'))['queries']
      if q['id'] in ('TR1', 'TO1', 'TO4', 'HV2', 'EP5', 'RM2')]


def toks(s):
    return len(s) / 3.5


def classify(line, seen_title=False):
    s = line.strip()
    if not s:
        return 'blank'
    if s.startswith('#') and re.match(r'#\d+', s):
        return 'header'
    if s.startswith('[') and '"' in s and not seen_title:
        return 'title_line'   # [type] "title" (...) — exactly once per block
    if s.startswith('['):
        return 'edges'        # edge lines also start with [type id:...]
    for prefix, comp in (
        ('Content:', 'content'), ('Situation:', 'situation'),
        ('Question:', 'question'), ('Reasoning:', 'reasoning'),
        ('Keywords:', 'keywords'), ('Created:', 'dates'),
        ('User Raw Quote:', 'quotes'), ('Anchor Raw Quote:', 'quotes'),
        ('Edges:', 'edges'), ('Correction Pattern:', 'corrections'),
    ):
        if s.startswith(prefix):
            return comp
    if s.startswith('⚠'):
        return 'corrections'
    if s.startswith('['):
        return 'edges'        # indented edge lines start with [type id:..]
    return 'continuation'     # wrapped text belonging to the previous field


with IsolatedBrain() as env:
    b = env.brain
    from servers.scales.s1.surface_contract import (
        format_candidate_for_surface, build_surface_prompt)

    agg = {}
    n_cands = 0
    prompt_level = []
    for q in QS:
        out = b.recall(query=q['query'], limit=25)
        raw = out.get('results', []) if isinstance(out, dict) else (out or [])
        # Mirror production (daemon_hooks.py:287): batch-enrich via brain.get_node
        # to rich shape (_metadata, _corrections, connections) + recall fields.
        # select_edges is skipped — render truncates to edge_limit=3 either way,
        # so token accounting is equivalent (edge CHOICE differs, cost doesn't).
        ids = [r.get('id', '') for r in raw]
        rich = b.get_node(ids)   # batch form returns {id: rich_node}
        rich_map = rich if isinstance(rich, dict) else {n.get('id'): n for n in [rich] if n}
        for nid, n in rich_map.items():
            if isinstance(n, dict) and 'id' not in n:
                n['id'] = nid
        cands = []
        for r in raw:
            nd = rich_map.get(r.get('id', '')) or dict(r)
            nd['score'] = r.get('effective_activation', 0)
            nd['discovery'] = r.get('_discovery', 'embedding')
            cands.append(nd)
        blocks = []
        for i, c in enumerate(cands, 1):
            txt = format_candidate_for_surface(c, i)
            blocks.append(txt)
            n_cands += 1
            prev = 'header'
            seen_title = False
            for line in txt.split('\n'):
                comp = classify(line, seen_title=seen_title)
                if comp == 'title_line':
                    seen_title = True
                if comp == 'continuation':
                    comp = prev
                elif comp != 'blank':
                    prev = comp
                agg[comp] = agg.get(comp, 0) + toks(line) + 0.3  # +newline
        cand_block = "\n\n".join(blocks)
        frame = ''
        try:
            ctx = b.get_or_create_session('recon-probe')
            frame = ctx.get_frame(b)
        except Exception as e:
            frame = ''
        user_content, max_tokens = build_surface_prompt(
            cands, q['query'], frame=frame)
        prompt_level.append({
            'id': q['id'], 'total': toks(user_content),
            'frame': toks(frame), 'candidates': toks(cand_block),
            'other': toks(user_content) - toks(frame) - toks(cand_block),
        })

    agg.pop('blank', None)
    total = sum(agg.values())
    print("\n=== per-candidate-block token distribution (%d candidates across %d queries) ==="
          % (n_cands, len(QS)))
    print("%-14s %10s %8s %12s" % ("component", "tokens", "share", "per-candidate"))
    for comp, t in sorted(agg.items(), key=lambda x: -x[1]):
        print("%-14s %10.0f %7.1f%% %12.1f" % (comp, t, 100*t/total, t/n_cands))
    print("%-14s %10.0f %8s %12.1f" % ("TOTAL", total, "100%", total/n_cands))

    print("\n=== assembled prompt level (user block only; system instructions separate) ===")
    print("%-5s %8s %8s %12s %8s" % ("query", "total", "frame", "candidates", "other"))
    for p in prompt_level:
        print("%-5s %8.0f %8.0f %12.0f %8.0f"
              % (p['id'], p['total'], p['frame'], p['candidates'], p['other']))
