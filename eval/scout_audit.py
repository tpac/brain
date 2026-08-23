"""Scout-output prevalence audit across a longmem eval run.

For every item in a run's artifacts dir:
  - Read traces.jsonl, pull scout_input + scout_findings for each scout.
  - Read result.json for axis/correct/failure_bucket.
  - Read nodes.jsonl + recall.json to check if gold tokens made it into
    any node title/content/AQ verbatim (lexical only — paraphrases miss).

Output a markdown table and a JSON dump:
  - one row per (item × scout) for the wide view
  - one row per item with scout-failure flags rolled up
  - prevalence summary: scout-silent-partial rate by axis, by correct/incorrect

Flag patterns:
  - SILENT_PARTIAL: considered > 0 AND passed_threshold == 0 AND no error
  - DISABLED: any error containing 'disabled'
  - EMPTY: scanned 0 turns (synthesis-scout-skip pattern)
  - ERROR: errors list non-empty

USE
---
    ./dev python3 eval/scout_audit.py \\
        eval/longmem/reports/eval_a_v15_6_2026_05_10 \\
        --out eval/longmem/reports/scout_audit_v15_6.md
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


SCOUT_NAMES = ['quote', 'temporal', 'facts', 'synthesis']


def _load_jsonl(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _scout_from_traces(traces):
    """Return {scout_name: {scanned_turns, considered, passed_threshold,
    candidates, errors, latency_ms}}."""
    out = {}
    for t in traces:
        ref = t.get('ref_type', '')
        if ref not in ('scout_input', 'scout_findings'):
            continue
        meta = t.get('metadata') or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        scout = meta.get('scout')
        if not scout:
            continue
        slot = out.setdefault(scout, {
            'scanned_turns': None, 'considered': None,
            'passed_threshold': None, 'candidates': None,
            'errors': None, 'latency_ms': None,
        })
        if ref == 'scout_input':
            scanned = meta.get('scanned') or {}
            if isinstance(scanned, dict):
                slot['scanned_turns'] = scanned.get('turns')
                # different scouts use different keys
                for k in ('phrases_considered', 'date_phrases_found',
                          'fact_claims_found', 'considered'):
                    if k in scanned:
                        slot['considered'] = scanned[k]
                        break
                slot['passed_threshold'] = scanned.get('passed_threshold')
            slot['latency_ms'] = meta.get('latency_ms')
        elif ref == 'scout_findings':
            slot['candidates'] = len(meta.get('candidate_handles') or [])
            slot['errors'] = meta.get('errors') or []
    return out


def _gold_terms(gold):
    """Lowercase distinctive tokens from gold answer (min 4 chars, no stops)."""
    if not gold:
        return []
    if not isinstance(gold, str):
        gold = str(gold)
    stop = {'the', 'a', 'an', 'and', 'or', 'but', 'so', 'this', 'that', 'with',
            'for', 'from', 'into', 'about', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'will', 'can', 'may', 'i', 'you', 'we', 'my',
            'your', 'our', 'they', 'them', 'their', 'it', 'its', 'in', 'on',
            'of', 'to', 'as', 'at', 'by', 'not', 'no', 'do', 'does', 'did',
            'said', 'says', 'mentioned', 'mention', 'enough', 'how', 'what',
            'when', 'where', 'who', 'which', 'why', 'according'}
    words = re.findall(r"[A-Za-z0-9$.]+", gold.lower())
    return [w for w in words if len(w) >= 4 and w not in stop]


def _check_gold_in_nodes(nodes, gold_terms):
    """Three states: 'verbatim' (5+ contiguous gold tokens in title/AQ),
    'paraphrase' (gold tokens appear scattered in content/AQ),
    'absent'."""
    if not gold_terms:
        return 'no_gold'
    # Verbatim: look for 5+ gold tokens in sequence in title or AQ
    for n in nodes:
        title = (n.get('title') or '').lower()
        kv = n.get('kv') or {}
        aq = (kv.get('my_raw_quote') or '').lower()
        uq = (kv.get('their_raw_quote') or '').lower()
        for blob in (title, aq, uq):
            # Count contiguous gold-term hits in tokenized blob
            tokens = re.findall(r"[a-z0-9$.]+", blob)
            run = 0
            best = 0
            for tok in tokens:
                if tok in gold_terms:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            if best >= 5:
                return 'verbatim'
    # Scattered presence — count distinct gold tokens hit anywhere in the
    # encoded node bodies. If >= 50% are present, call it paraphrase.
    gold_set = set(gold_terms)
    hit_set = set()
    for n in nodes:
        blob = ' '.join([
            (n.get('title') or ''),
            (n.get('content') or ''),
            ((n.get('kv') or {}).get('my_raw_quote') or ''),
            ((n.get('kv') or {}).get('their_raw_quote') or ''),
            ((n.get('kv') or {}).get('situation') or ''),
            ((n.get('kv') or {}).get('reasoning') or ''),
        ]).lower()
        tokens = set(re.findall(r"[a-z0-9$.]+", blob))
        hit_set |= (gold_set & tokens)
    coverage = len(hit_set) / max(1, len(gold_set))
    if coverage >= 0.5:
        return 'paraphrase'
    elif coverage > 0:
        return 'partial'
    return 'absent'


def _flag(scout_data):
    """Return list of flag strings for one scout dict."""
    if scout_data is None:
        return ['MISSING']
    flags = []
    errs = scout_data.get('errors') or []
    if any('disabled' in str(e).lower() for e in errs):
        flags.append('DISABLED')
    elif errs:
        flags.append('ERROR')
    scanned = scout_data.get('scanned_turns')
    if scanned == 0:
        flags.append('EMPTY')
    considered = scout_data.get('considered') or 0
    passed = scout_data.get('passed_threshold')
    cands = scout_data.get('candidates') or 0
    if considered and considered > 0 and (passed == 0 or cands == 0) and not errs:
        flags.append('SILENT_PARTIAL')
    return flags or ['OK']


def audit_run(run_dir: Path):
    items_dir = run_dir / 'items'
    if not items_dir.is_dir():
        raise RuntimeError(f'no items/ in {run_dir}')

    rows = []
    for qdir in sorted(items_dir.iterdir()):
        if not qdir.is_dir():
            continue
        qid = qdir.name
        result = {}
        try:
            result = json.loads((qdir / 'result.json').read_text())
        except Exception:
            continue
        meta = {}
        try:
            meta = json.loads((qdir / 'meta.json').read_text())
        except Exception:
            pass
        traces = _load_jsonl(qdir / 'traces.jsonl')
        nodes = _load_jsonl(qdir / 'nodes.jsonl')

        scout_map = _scout_from_traces(traces)
        gold = result.get('answer_gold') or meta.get('gold') or ''
        if isinstance(gold, list):
            gold = ' '.join(str(g) for g in gold)
        gold_terms = _gold_terms(gold)
        enc_state = _check_gold_in_nodes(nodes, gold_terms)

        per_scout = {}
        for name in SCOUT_NAMES:
            sd = scout_map.get(name)
            per_scout[name] = {
                'data': sd,
                'flags': _flag(sd),
            }

        rows.append({
            'qid': qid,
            'axis': result.get('axis') or meta.get('axis'),
            'correct': result.get('correct'),
            'failure_bucket': result.get('failure_bucket'),
            'haystack_turn_count': meta.get('haystack_turn_count'),
            'node_count': len(nodes),
            'gold_encoded': enc_state,
            'gold_terms_count': len(gold_terms),
            'scouts': per_scout,
        })
    return rows


def render_markdown(rows, run_name):
    lines = []
    lines.append(f'# Scout-output audit — {run_name}')
    lines.append('')
    lines.append(f'**Total items:** {len(rows)}')
    lines.append('')

    # ─── Prevalence summary ──────────────────────────────────────
    lines.append('## Prevalence summary')
    lines.append('')
    flag_totals = defaultdict(lambda: defaultdict(int))
    silent_partial_by_axis = defaultdict(lambda: {'pass': 0, 'fail': 0,
                                                    'pass_sp': 0, 'fail_sp': 0})
    for r in rows:
        axis = r['axis'] or 'unknown'
        correct = r['correct']
        any_silent_partial = False
        for name in SCOUT_NAMES:
            flags = r['scouts'][name]['flags']
            for f in flags:
                flag_totals[name][f] += 1
            if 'SILENT_PARTIAL' in flags:
                any_silent_partial = True
        bucket = 'pass' if correct else 'fail'
        silent_partial_by_axis[axis][bucket] += 1
        if any_silent_partial:
            silent_partial_by_axis[axis][bucket + '_sp'] += 1

    # Flag counts per scout
    lines.append('### Flags per scout (count of items)')
    lines.append('')
    lines.append('| Scout | OK | SILENT_PARTIAL | EMPTY | ERROR | DISABLED | MISSING |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for name in SCOUT_NAMES:
        ft = flag_totals[name]
        lines.append(f"| {name} | {ft.get('OK',0)} | {ft.get('SILENT_PARTIAL',0)} | "
                     f"{ft.get('EMPTY',0)} | {ft.get('ERROR',0)} | {ft.get('DISABLED',0)} | "
                     f"{ft.get('MISSING',0)} |")
    lines.append('')

    # SILENT_PARTIAL rate by axis × pass/fail
    lines.append('### SILENT_PARTIAL rate by axis × outcome')
    lines.append('')
    lines.append('| Axis | Pass (n) | Pass w/ SP | Fail (n) | Fail w/ SP |')
    lines.append('|---|---:|---:|---:|---:|')
    for axis in sorted(silent_partial_by_axis.keys()):
        d = silent_partial_by_axis[axis]
        p_pct = f"{d['pass_sp']}/{d['pass']}" if d['pass'] else "—"
        f_pct = f"{d['fail_sp']}/{d['fail']}" if d['fail'] else "—"
        lines.append(f'| {axis} | {d["pass"]} | {p_pct} | {d["fail"]} | {f_pct} |')
    lines.append('')

    # Gold-encoded state distribution
    lines.append('### Gold-fact encoded state')
    lines.append('')
    lines.append('| Outcome | verbatim | paraphrase | partial | absent | no_gold |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for bucket in ('pass', 'fail'):
        counts = defaultdict(int)
        for r in rows:
            if (r['correct'] and bucket == 'pass') or (not r['correct'] and bucket == 'fail'):
                counts[r['gold_encoded']] += 1
        lines.append(f"| {bucket} | {counts.get('verbatim',0)} | {counts.get('paraphrase',0)} | "
                     f"{counts.get('partial',0)} | {counts.get('absent',0)} | "
                     f"{counts.get('no_gold',0)} |")
    lines.append('')

    # ─── Wide table ──────────────────────────────────────────────
    lines.append('## Per-item table')
    lines.append('')
    hdr = ['qid', 'axis', '✓', 'bucket', 'gold-enc', 'nodes']
    for name in SCOUT_NAMES:
        hdr.append(f'{name}-flags')
        hdr.append(f'{name}-c/p/cand')
    lines.append('| ' + ' | '.join(hdr) + ' |')
    lines.append('|' + '|'.join(['---'] * len(hdr)) + '|')
    for r in sorted(rows, key=lambda x: (x['correct'] or False, x['axis'], x['qid'])):
        cells = [
            r['qid'],
            r['axis'] or '',
            '✓' if r['correct'] else '✗',
            r['failure_bucket'] or '-',
            r['gold_encoded'],
            str(r['node_count']),
        ]
        for name in SCOUT_NAMES:
            s = r['scouts'][name]
            cells.append(','.join(s['flags']))
            d = s['data'] or {}
            cells.append(f"{d.get('considered','-')}/{d.get('passed_threshold','-')}/{d.get('candidates','-')}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    # ─── SILENT_PARTIAL deep-list ────────────────────────────────
    lines.append('## SILENT_PARTIAL items (full list, fail first)')
    lines.append('')
    sp_items = []
    for r in rows:
        sp_scouts = [name for name in SCOUT_NAMES
                     if 'SILENT_PARTIAL' in r['scouts'][name]['flags']]
        if sp_scouts:
            sp_items.append((r, sp_scouts))
    sp_items.sort(key=lambda t: (t[0]['correct'] or False, t[0]['qid']))
    for r, sp_scouts in sp_items:
        marker = '✓' if r['correct'] else '✗'
        lines.append(f"- {marker} `{r['qid']}` axis={r['axis']} "
                     f"bucket={r['failure_bucket'] or '-'} "
                     f"gold-enc={r['gold_encoded']} "
                     f"silent_partial=[{','.join(sp_scouts)}]")
    lines.append('')

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', help='Path to eval_a_X_run dir (containing items/)')
    p.add_argument('--out', default=None)
    p.add_argument('--json-out', default=None)
    args = p.parse_args()
    rd = Path(args.run_dir)
    rows = audit_run(rd)
    run_name = rd.name
    md = render_markdown(rows, run_name)
    out = args.out or str(rd / 'scout_audit.md')
    Path(out).write_text(md)
    print(f'[audit] wrote {out}')
    json_out = args.json_out or str(rd / 'scout_audit.json')
    Path(json_out).write_text(json.dumps(rows, indent=2, default=str))
    print(f'[audit] raw → {json_out}')


if __name__ == '__main__':
    main()
