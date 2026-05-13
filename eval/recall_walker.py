"""Deep recall analyzer for v5 agentic eval runs.

For each completed item, walks:
  L0 — Encoder: was the gold-bearing content encoded?
  L1 — Pre-seed: did cosine put it in the 25 candidates?
  L2 — Tool routing: which tools did Haiku call, what did they return?
  L3 — Selection: did Haiku pick the right candidate(s)?
  L4 — Render: did the content reach the answerer?
  L5 — Answer: did the answerer use it?

Outputs a markdown report joining all six layers per item.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


STOP = {'the','about','that','what','with','from','this','have','were','was',
        'said','your','their','they','will','would','these','those','some',
        'when','where','which','there','here','then','than','only','just',
        'mentioned','prior','previous','conversation'}


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def jlload(p):
    out = []
    try:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except Exception:
        pass
    return out


def gold_terms(gold):
    if not isinstance(gold, str):
        gold = str(gold)
    words = re.findall(r'[A-Za-z0-9$.]+', gold.lower())
    return [w for w in words if len(w) >= 4 and w not in STOP]


def best_gold_node(nodes, gset):
    best = (None, None, 0, '')
    if not gset:
        return best
    for n in nodes:
        title = n.get('title') or ''
        content = n.get('content') or ''
        kv = n.get('kv') or {}
        blob = ' '.join([title, content,
                         kv.get('anchor_raw_quote', '') or '',
                         kv.get('user_raw_quote', '') or '',
                         kv.get('situation', '') or '',
                         kv.get('reasoning', '') or '']).lower()
        toks = set(re.findall(r'[a-z0-9$.]+', blob))
        hits = len(gset & toks)
        if hits > best[2]:
            # excerpt around first hit
            excerpt = ''
            for g in gset:
                idx = blob.find(g)
                if idx >= 0:
                    s = max(0, idx - 50)
                    e = min(len(blob), idx + 180)
                    excerpt = '...' + blob[s:e].replace('\n', ' ') + '...'
                    break
            best = (n.get('id'), title, hits, excerpt)
    return best


def analyze_item(item_dir: Path) -> dict:
    qid = item_dir.name
    meta = jload(item_dir / 'meta.json', {}) or {}
    result = jload(item_dir / 'result.json', {}) or {}
    recall = jload(item_dir / 'recall.json', {}) or {}
    traces = jlload(item_dir / 'traces.jsonl')
    nodes = jlload(item_dir / 'nodes.jsonl')

    gold = str(meta.get('gold', ''))
    gset = set(gold_terms(gold))

    # L0: Encoder
    enc_id, enc_title, enc_hits, enc_excerpt = best_gold_node(nodes, gset)
    # Calibrate to gold-term count: a 3-hit "match" on a 50-term gold is noise;
    # a 3-hit "match" on a 4-term gold is real coverage.
    if not gset:
        enc_state = 'no_gold_terms'
    else:
        coverage = enc_hits / max(1, len(gset))
        if coverage >= 0.7 and enc_hits >= 4:
            enc_state = 'verbatim_or_strong'
        elif coverage >= 0.4 and enc_hits >= 3:
            enc_state = 'paraphrase'
        elif coverage >= 0.25 or enc_hits >= 3:
            enc_state = 'partial'
        else:
            enc_state = 'absent'  # weak token overlap, treat as missing

    # L1: Pre-seed
    candidates = recall.get('candidates') or []
    cand_count = recall.get('candidate_count', len(candidates))
    enc_rank = None
    if enc_id:
        for i, c in enumerate(candidates):
            cid = c.get('id') if isinstance(c, dict) else c
            if cid == enc_id:
                enc_rank = i + 1
                break

    # L2: Tool routing
    surface_meta = None
    for t in traces:
        if t.get('ref_type') == 'surface_selected':
            m = t.get('metadata') or {}
            if isinstance(m, str):
                try:
                    m = json.loads(m)
                except Exception:
                    m = {}
            surface_meta = m
            break

    tool_calls = []
    surface_variant = '(unknown)'
    if surface_meta:
        surface_variant = surface_meta.get('surface_variant', 'unknown')
        for r in surface_meta.get('tool_trace') or []:
            for tc in r.get('tool_calls') or []:
                tool_calls.append({
                    'round': r.get('round'),
                    'tool': tc.get('tool'),
                    'args': tc.get('args') or {},
                    'result_count': tc.get('result_count', 0),
                    'latency_ms': tc.get('latency_ms', 0),
                    'error': tc.get('error'),
                })

    # L3: Selection
    selected = recall.get('selected') or []
    selected_ids = []
    selected_modes = []
    for s in selected:
        if isinstance(s, dict):
            selected_ids.append(s.get('id'))
            selected_modes.append(s.get('mode', 'arc'))
        else:
            selected_ids.append(s)
            selected_modes.append('?')
    enc_in_selected = enc_id in selected_ids if enc_id else False

    # L4: Render — was gold-bearing node's content rendered?
    ctx = recall.get('context', '') or ''
    enc_content_in_ctx = False
    if enc_id:
        short_id = enc_id[:8]
        # Look for "(id:<short>" near a "Content:" header
        pattern = re.compile(r'\(id:' + re.escape(short_id) +
                              r'[^)]*\)[\s\S]{0,400}?Content:',
                              re.IGNORECASE)
        enc_content_in_ctx = bool(pattern.search(ctx))

    # L5: Answerer
    hypothesis = result.get('hypothesis', '') or ''
    abstained = result.get('abstained', None)
    has_context = result.get('has_context', None)
    correct = result.get('correct', None)
    bucket = result.get('failure_bucket', '-')

    # Layer-of-failure call
    if correct:
        layer = 'PASS'
    elif enc_state == 'absent':
        layer = 'L0_ENCODER_MISSED'
    elif enc_state == 'partial':
        layer = 'L0_ENCODER_PARTIAL'
    elif enc_rank is None and enc_id is not None:
        layer = 'L1_PRESEED_MISSED (gold not in cosine top-N)'
    elif not enc_in_selected and enc_id is not None:
        layer = 'L3_SURFACE_SKIPPED'
    elif not enc_content_in_ctx and enc_id is not None:
        layer = 'L4_RENDER_DROPPED'
    elif correct is False:
        layer = 'L5_ANSWERER'
    else:
        layer = 'PENDING'

    return {
        'qid': qid,
        'axis': meta.get('axis') or result.get('axis'),
        'question': meta.get('question', '')[:200],
        'gold': gold[:200],
        'gold_terms': len(gset),
        'enc_state': enc_state,
        'enc_id': enc_id[:8] if enc_id else None,
        'enc_title': (enc_title or '')[:80],
        'enc_excerpt': enc_excerpt,
        'enc_hits': enc_hits,
        'cand_count': cand_count,
        'enc_rank': enc_rank,
        'surface_variant': surface_variant,
        'tool_calls': tool_calls,
        'selected_count': len(selected_ids),
        'selected_modes': selected_modes,
        'enc_in_selected': enc_in_selected,
        'enc_content_in_ctx': enc_content_in_ctx,
        'context_chars': len(ctx),
        'hypothesis': hypothesis[:200],
        'abstained': abstained,
        'has_context': has_context,
        'correct': correct,
        'bucket': bucket,
        'layer': layer,
    }


def render_markdown(reports, run_name):
    lines = [f'# Recall walker — {run_name}', '']
    from collections import Counter
    layer_counts = Counter(r['layer'] for r in reports)
    lines.append(f'**Total analyzed:** {len(reports)}')
    lines.append('')
    lines.append('## Layer-of-failure roll-up')
    lines.append('')
    lines.append('| Layer | Count |')
    lines.append('|---|---:|')
    for layer, n in layer_counts.most_common():
        lines.append(f'| `{layer}` | {n} |')
    lines.append('')

    # Tool usage roll-up
    from collections import defaultdict
    tool_usage = defaultdict(int)
    tool_with_results = defaultdict(int)
    items_with_tools = 0
    for r in reports:
        if r['tool_calls']:
            items_with_tools += 1
        for tc in r['tool_calls']:
            tool_usage[tc['tool']] += 1
            if tc.get('result_count', 0) > 0:
                tool_with_results[tc['tool']] += 1
    lines.append('## Tool usage')
    lines.append('')
    lines.append(f'**Items with any tool call:** {items_with_tools}/{len(reports)}')
    lines.append('')
    lines.append('| Tool | Calls | Calls returning results |')
    lines.append('|---|---:|---:|')
    for tool in sorted(tool_usage.keys()):
        lines.append(f'| `{tool}` | {tool_usage[tool]} | {tool_with_results[tool]} |')
    lines.append('')

    # Per-item detail
    lines.append('## Per-item walk')
    lines.append('')
    for r in sorted(reports, key=lambda x: (x['correct'] is True, x['qid'])):
        verdict = '✓' if r['correct'] else ('✗' if r['correct'] is False else '?')
        lines.append(f"### {verdict} `{r['qid']}` — {r['axis']} — {r['layer']}")
        lines.append('')
        lines.append(f"- **Q:** {r['question']}")
        lines.append(f"- **Gold:** {r['gold']}")
        lines.append(f"- **Hypothesis:** {r['hypothesis']}")
        lines.append(f"- **Classifier bucket:** `{r['bucket']}`")
        lines.append('')
        lines.append('**L0 Encoder:**')
        if r['enc_id']:
            lines.append(f"- best gold-bearing node: `{r['enc_id']}` — *{r['enc_title']}*")
            lines.append(f"- state: `{r['enc_state']}` ({r['enc_hits']}/{r['gold_terms']} terms)")
            if r['enc_excerpt']:
                lines.append(f"- excerpt: {r['enc_excerpt']}")
        else:
            lines.append(f"- state: `{r['enc_state']}` (no candidate node carries gold terms)")
        lines.append('')
        lines.append(f"**L1 Pre-seed:** {r['cand_count']} candidates, gold-node rank: "
                     f"**{r['enc_rank'] or 'NOT IN POOL'}**")
        lines.append('')
        lines.append(f"**L2 Tool routing:** variant=`{r['surface_variant']}`, {len(r['tool_calls'])} calls")
        if r['tool_calls']:
            for tc in r['tool_calls']:
                args = tc['args']
                args_short = ', '.join(f"{k}={str(v)[:30]}" for k, v in list(args.items())[:2])
                err = (' ERR=' + tc['error']) if tc.get('error') else ''
                lines.append(f"  - R{tc['round']} `{tc['tool']}({args_short})` → "
                             f"{tc['result_count']} results, {tc['latency_ms']}ms{err}")
        lines.append('')
        lines.append(f"**L3 Selection:** {r['selected_count']} chosen, "
                     f"gold in selected: **{r['enc_in_selected']}**, "
                     f"modes: {r['selected_modes']}")
        lines.append('')
        lines.append(f"**L4 Render:** content rendered: **{r['enc_content_in_ctx']}**, "
                     f"context: {r['context_chars']} chars")
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--only-completed', action='store_true',
                   help='Skip items without result.json')
    args = p.parse_args()
    rd = Path(args.run_dir)
    items_dir = rd / 'items'
    reports = []
    for d in sorted(items_dir.iterdir()):
        if not d.is_dir():
            continue
        if args.only_completed and not (d / 'result.json').exists():
            continue
        if not (d / 'recall.json').exists():
            # not far enough through pipeline
            continue
        reports.append(analyze_item(d))
    out = args.out or str(rd / 'recall_walk.md')
    Path(out).write_text(render_markdown(reports, rd.name))
    print(f'wrote {out} ({len(reports)} items)')
    Path(out.replace('.md', '.json')).write_text(
        json.dumps(reports, indent=2, default=str))


if __name__ == '__main__':
    main()
