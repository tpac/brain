"""Walk every failure item in an eval run and produce a layered diagnosis.

For each failure, classify the failure layer:
  - ENCODER: gold fact not in any node body
  - RECALL_RANK: gold-bearing node NOT in top candidates
  - SURFACE_PICK: gold-bearing node in candidates but not selected
  - CONTEXT_RENDER: selected but its content didn't reach the answerer
  - ANSWERER: content in context but answerer abstained/wrong
  - SCOUT_SILENT_PARTIAL: quote scout dropped phrases that mattered (annotation)

The layered approach catches the bug found in edced276:
  selected=2 IDs but only 1 rendered fully, the other became edge mention only.

Outputs markdown report with one block per item.
"""
import argparse
import json
import re
import sys
from pathlib import Path


def _load_jsonl(path):
    if not path.exists(): return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try:
            out.append(json.loads(line))
        except: pass
    return out


def _gold_terms(gold):
    if not isinstance(gold, str): gold = str(gold)
    stop = {'the','a','an','and','or','but','so','this','that','with','for','from',
            'into','about','is','are','was','were','be','have','has','had','will',
            'can','may','i','you','we','my','your','our','they','them','their','it',
            'its','in','on','of','to','as','at','by','not','no','do','does','did',
            'said','says','mentioned','mention','enough','how','what','when','where',
            'who','which','why','according','total','some'}
    words = re.findall(r"[A-Za-z0-9$.]+", gold.lower())
    return [w for w in words if len(w) >= 4 and w not in stop]


def _find_gold_in_nodes(nodes, gold_terms):
    """Return (best_node_id, best_node_title, encoding_state, blob_excerpt)."""
    if not gold_terms:
        return None, None, 'no_gold_terms', ''
    gold_set = set(gold_terms)
    best = (None, None, 0, '')
    for n in nodes:
        title = n.get('title','') or ''
        content = n.get('content','') or ''
        kv = n.get('kv') or {}
        blob = ' '.join([title, content,
                          kv.get('my_raw_quote',''),
                          kv.get('their_raw_quote',''),
                          kv.get('situation',''),
                          kv.get('reasoning','')]).lower()
        tokens = set(re.findall(r"[a-z0-9$.]+", blob))
        hits = len(gold_set & tokens)
        if hits > best[2]:
            # Find excerpt around first gold term
            excerpt = ''
            for term in gold_terms:
                idx = blob.find(term)
                if idx >= 0:
                    s = max(0, idx-60)
                    e = min(len(blob), idx+200)
                    excerpt = '...' + blob[s:e] + '...'
                    break
            best = (n.get('id'), title, hits, excerpt)
    if best[2] >= 5:
        return best[0], best[1], 'verbatim_or_strong', best[3]
    elif best[2] >= 3:
        return best[0], best[1], 'paraphrase', best[3]
    elif best[2] >= 1:
        return best[0], best[1], 'partial', best[3]
    return None, None, 'absent', ''


def _scout_summary(traces):
    out = {}
    for t in traces:
        if t.get('ref_type') != 'scout_findings':
            continue
        meta = t.get('metadata') or {}
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except: meta = {}
        scout = meta.get('scout','?')
        cands = meta.get('candidate_handles') or []
        errs = meta.get('errors') or []
        out[scout] = {'cands': len(cands), 'errors': errs}
    for t in traces:
        if t.get('ref_type') != 'scout_input':
            continue
        meta = t.get('metadata') or {}
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except: meta = {}
        scout = meta.get('scout','?')
        if scout in out:
            scanned = meta.get('scanned') or {}
            if isinstance(scanned, dict):
                considered = (scanned.get('phrases_considered') or
                              scanned.get('date_phrases_found') or
                              scanned.get('fact_claims_found') or
                              scanned.get('considered'))
                out[scout]['considered'] = considered
                out[scout]['passed'] = scanned.get('passed_threshold')
    return out


def _diagnose(item_dir: Path):
    """Returns dict with layered diagnosis."""
    try:
        result = json.loads((item_dir/'result.json').read_text())
        meta = json.loads((item_dir/'meta.json').read_text())
    except Exception as e:
        return {'qid': item_dir.name, 'error': str(e)}
    nodes = _load_jsonl(item_dir/'nodes.jsonl')
    traces = _load_jsonl(item_dir/'traces.jsonl')
    try:
        recall = json.loads((item_dir/'recall.json').read_text())
    except:
        recall = {}

    qid = item_dir.name
    axis = result.get('axis')
    correct = result.get('correct')
    gold = result.get('answer_gold')
    gold_str = str(gold)
    hypothesis = result.get('hypothesis','')
    bucket = result.get('failure_bucket') or '-'

    gold_terms = _gold_terms(gold_str)
    enc_id, enc_title, enc_state, enc_excerpt = _find_gold_in_nodes(nodes, gold_terms)

    # Recall layer
    candidates = recall.get('candidates') or []
    selected = recall.get('selected') or []
    selected_ids = [(s.get('id') if isinstance(s, dict) else s) for s in selected]
    cand_count = recall.get('candidate_count', len(candidates))

    enc_rank = None
    if enc_id:
        for i, c in enumerate(candidates):
            cid = c.get('id') if isinstance(c, dict) else c
            if cid == enc_id:
                enc_rank = i + 1
                break

    enc_in_selected = enc_id in selected_ids if enc_id else False

    # Context layer — was the gold-bearing node's CONTENT rendered?
    ctx = recall.get('context','') or ''
    enc_content_in_ctx = False
    enc_short_id = enc_id[:8] if enc_id else ''
    if enc_short_id:
        # Check if node ID appears as a (id:XXXX) standalone section header
        pattern = re.compile(r'\(id:' + re.escape(enc_short_id) + r'[^)]*\)\s*\n\s*Content:', re.IGNORECASE)
        enc_content_in_ctx = bool(pattern.search(ctx))

    # Layer decision
    if enc_state == 'absent':
        layer = 'L2:ENCODER_MISSED'
    elif enc_state == 'partial':
        # Encoder caught some tokens but not the load-bearing ones
        if not enc_id:
            layer = 'L2:ENCODER_PARTIAL'
        elif enc_rank is None:
            layer = 'L3:RECALL_BURIED'
        elif not enc_in_selected:
            layer = 'L4:SURFACE_SKIPPED'
        else:
            layer = 'L2:ENCODER_PARTIAL'
    elif enc_state == 'paraphrase':
        if enc_rank is None:
            layer = 'L3:RECALL_BURIED (paraphrased node not surfaced)'
        elif not enc_in_selected:
            layer = 'L4:SURFACE_SKIPPED'
        elif not enc_content_in_ctx:
            layer = 'L5:CONTEXT_RENDER_DROPPED'
        else:
            layer = 'L6:ANSWERER (paraphrased context wasn\'t enough)'
    elif enc_state == 'verbatim_or_strong':
        if enc_rank is None:
            layer = 'L3:RECALL_BURIED (gold-bearing node not in top-N)'
        elif not enc_in_selected:
            layer = 'L4:SURFACE_SKIPPED'
        elif not enc_content_in_ctx:
            layer = 'L5:CONTEXT_RENDER_DROPPED'
        else:
            layer = 'L6:ANSWERER'
    else:
        layer = 'UNKNOWN'

    scouts = _scout_summary(traces)

    return {
        'qid': qid,
        'axis': axis,
        'correct': correct,
        'bucket_classifier': bucket,
        'layer_diagnosis': layer,
        'gold': gold_str[:200],
        'hypothesis': hypothesis[:200],
        'encoded_state': enc_state,
        'encoded_node_id': enc_id[:8] if enc_id else None,
        'encoded_node_title': (enc_title or '')[:90],
        'encoded_excerpt': enc_excerpt[:300],
        'gold_term_count': len(gold_terms),
        'cand_count': cand_count,
        'enc_rank_in_candidates': enc_rank,
        'enc_in_selected': enc_in_selected,
        'enc_content_in_context': enc_content_in_ctx,
        'selected_count': len(selected_ids),
        'context_chars': len(ctx),
        'scouts': scouts,
        'node_count': len(nodes),
    }


def render(diagnoses, run_name):
    lines = []
    lines.append(f'# Layered failure diagnosis — {run_name}')
    lines.append('')

    # ─── Layer-bucket roll-up ────────────────────────────────────
    from collections import Counter
    layer_counts = Counter(d['layer_diagnosis'].split(' ')[0] for d in diagnoses
                            if not d['correct'])
    lines.append('## Failure roll-up by layer')
    lines.append('')
    lines.append('| Layer | Count | Note |')
    lines.append('|---|---:|---|')
    for layer, n in layer_counts.most_common():
        lines.append(f'| `{layer}` | {n} | |')
    lines.append('')

    # ─── Classifier vs new layer diagnosis ───────────────────────
    lines.append('## Classifier bucket vs layer diagnosis (failures only)')
    lines.append('')
    lines.append('| qid | axis | classifier | layer | encoded_state |')
    lines.append('|---|---|---|---|---|')
    for d in diagnoses:
        if d['correct']: continue
        lines.append(f"| `{d['qid']}` | {d['axis']} | {d['bucket_classifier']} | "
                     f"{d['layer_diagnosis']} | {d['encoded_state']} |")
    lines.append('')

    # ─── Per-item detail ─────────────────────────────────────────
    lines.append('## Per-item diagnosis')
    lines.append('')
    for d in diagnoses:
        if d['correct']: continue
        s = d['scouts']
        scout_line = ' | '.join(
            f"{name}: c={s.get(name,{}).get('considered','-')} p={s.get(name,{}).get('passed','-')} cands={s.get(name,{}).get('cands','-')}"
            for name in ['quote','temporal','facts','synthesis'])
        lines.append(f"### `{d['qid']}` — {d['axis']} — {d['layer_diagnosis']}")
        lines.append('')
        lines.append(f"- **Gold:** {d['gold']}")
        lines.append(f"- **Hypothesis:** {d['hypothesis']}")
        lines.append(f"- **Classifier bucket:** `{d['bucket_classifier']}`")
        lines.append(f"- **Encoded state:** `{d['encoded_state']}` "
                     f"(gold terms hit in best node: ~{d['encoded_state']})")
        if d['encoded_node_id']:
            lines.append(f"- **Best gold-matching node:** `{d['encoded_node_id']}` — "
                         f"*{d['encoded_node_title']}*")
            if d['encoded_excerpt']:
                lines.append(f"  - excerpt: {d['encoded_excerpt']}")
        lines.append(f"- **Recall:** {d['cand_count']} candidates, "
                     f"gold-node rank: **{d['enc_rank_in_candidates'] or 'NOT IN TOP-N'}**, "
                     f"selected_count={d['selected_count']}")
        lines.append(f"- **Surface:** gold-node in selected? **{d['enc_in_selected']}** "
                     f"| content rendered to answerer? **{d['enc_content_in_context']}** "
                     f"(ctx={d['context_chars']} chars)")
        lines.append(f"- **Scouts:** {scout_line}")
        lines.append('')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--include-passes', action='store_true')
    args = p.parse_args()
    rd = Path(args.run_dir)
    items_dir = rd / 'items'
    diagnoses = []
    for qdir in sorted(items_dir.iterdir()):
        if not qdir.is_dir(): continue
        d = _diagnose(qdir)
        if 'error' in d: continue
        if not args.include_passes and d['correct']: continue
        diagnoses.append(d)
    out = args.out or str(rd / 'failure_diagnosis.md')
    Path(out).write_text(render(diagnoses, rd.name))
    json_out = out.replace('.md', '.json')
    Path(json_out).write_text(json.dumps(diagnoses, indent=2, default=str))
    print(f'wrote {out} ({len(diagnoses)} items)')


if __name__ == '__main__':
    main()
