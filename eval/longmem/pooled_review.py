"""Pooled-brain turn-by-turn review — the §20.18 smoke-approval package.

Renders everything the operator needs to approve a pooled build before
scaling: per session, per turn — the conversation, what recall surfaced
(the S1R judge output riding get_conversation), what the Scribe encoded
(every non-seed node with situation + source_refs), the S2 deltas, the V0
audit, the per-item gold scans, and an index of the captured S1E prompts
(one full payload inlined as the representative example).

Read-only over a WORKING COPY of the frozen corpus dir (the sweep.py
pattern) — the frozen brain is never opened in place. All trace reads go
through the brain query API (get_conversation), never TraceDAL/raw SQL.

Run:  ./dev python3 eval/longmem/pooled_review.py --corpus <hash>
      writes eval/longmem/reports/pooled_review_<hash>.md
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.longmem.build_corpus import _pooled_session_plan, _load_env
from eval.longmem.corpus import corpus_dir, load_manifest
from eval.longmem.fresh_brain import create_fresh_eval_brain

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')


def _trim(text, n=400):
    text = (text or '').replace('\n', ' ').strip()
    return text if len(text) <= n else text[:n] + ' …[%d chars]' % len(text)


def render(corpus_hash: str) -> str:
    m = load_manifest(corpus_hash)
    if not m or not m.get('config', {}).get('pooled'):
        raise SystemExit('corpus %s is not a pooled corpus (or has no manifest)'
                         % corpus_hash)
    audit = m.get('pooled_audit', {})
    qids = m['config']['qids']

    # Reconstruct the session plan (same function the build used — the ids
    # and order are derived, not stored, so they can't drift from the build).
    with open(os.path.join('eval/longmem/data', m['config']['oracle'])) as f:
        oracle = json.load(f)
    by_id = {it['question_id']: it for it in oracle}
    plan = _pooled_session_plan([by_id[q] for q in qids])

    # Working copy — never open the frozen brain in place.
    pooled_dir = m['items'][0]['brain_dir']
    work = os.path.join('/tmp', 'pooled-review-%s' % corpus_hash)
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(pooled_dir, work)
    brain = create_fresh_eval_brain(path=work, wipe=False)

    out = []
    w = out.append
    w('# Pooled review — corpus %s (%s)\n' % (corpus_hash, m.get('label')))
    w('## V0 audit\n')
    w('| check | value |\n|---|---|')
    for k in ('sessions', 'user_turns', 'user_turns_replayed',
              'dates_monotonic', 'node_count', 'span', 'green'):
        w('| %s | %s |' % (k, audit.get(k)))
    w('| build errors | %s |' % audit.get('build_errors', {}).get('count'))
    w('| build wall | %.1fs |' % (m.get('build_ms', 0) / 1000.0))
    w('| s1e/s2 runs | %s / %s |\n' % (
        m.get('pooled_totals', {}).get('s1e_runs'),
        m.get('pooled_totals', {}).get('s2_runs')))

    w('## Gold scans on the POOLED brain (vs per-item history)\n')
    for it in m['items']:
        gs = it['gold_scan']
        w('- **%s** (%s): %s — terms=%s matches=%s' % (
            it['qid'], it['axis'],
            'ANSWERABLE' if it['answerable'] else 'UNANSWERABLE',
            gs.get('terms_used'), [x for x in (gs.get('matches') or [])][:4]))
    w('')

    # ── Turn-by-turn: conversation + surface picks, via the ONE door ──
    for i, e in enumerate(plan):
        sid = e['sid']
        w('\n---\n## Session %d/%d — `%s` (%s)\n' % (
            i + 1, len(plan), sid, e['date']))
        turns = brain.get_conversation(sid, limit=100, with_judge_output=True)
        if not turns:
            w('**⚠ NO TRACED TURNS — investigate before approving.**')
            continue
        t_no = 0
        for t in turns:
            if t['role'] == 'user':
                t_no += 1
                w('**T%d op** (`trace:%s`): %s' % (
                    t_no, t.get('trace_id'), _trim(t['content'])))
                jo = t.get('judge_output')
                if jo:
                    w('  - *surfaced:* %s' % _trim(str(jo), 500))
                else:
                    w('  - *surfaced:* (nothing injected)')
            else:
                w('  - **anchor**: %s' % _trim(t['content'], 300))
        w('')

    # ── Encoded nodes: everything non-seed, grouped by encoding_source ──
    w('\n---\n## Encoded nodes (non-seed)\n')
    rows = brain.filter_nodes(field='encoding_source',
                              exclude=['seed'], limit=200, rich=False)
    nodes = rows.get('nodes', rows) if isinstance(rows, dict) else rows
    by_src = {}
    for nd in nodes:
        by_src.setdefault(nd.get('encoding_source') or '?', []).append(nd)
    for src in sorted(by_src):
        w('\n### %s (%d)\n' % (src, len(by_src[src])))
        for nd in by_src[src]:
            full = brain.get_node(nd['id'])
            node = (full or {}).get('node', full) or nd
            w('- `%s` [%s] **%s**' % (nd['id'], nd.get('type'),
                                      nd.get('title')))
            sit = node.get('situation') or (node.get('metadata') or {}).get('situation')
            if sit:
                w('    - situation: %s' % _trim(str(sit), 220))
            refs = node.get('source_refs') or (node.get('metadata') or {}).get('source_refs')
            if refs:
                w('    - source_refs: %s' % refs)

    # ── Prompts index + one representative payload ──
    # prompts_dir points at the pooled brain's payloads/ root (recorder
    # layout: payloads/{date}/{chain}/NNN-round_payload.json) — walk it.
    pdir = m.get('prompts_dir') or os.path.join(corpus_dir(corpus_hash), 'prompts')
    w('\n---\n## Captured S1E prompts (`%s`)\n' % pdir)
    files = []
    if os.path.isdir(pdir):
        for root_dir, _dirs, names in os.walk(pdir):
            for fn in sorted(names):
                files.append(os.path.join(root_dir, fn))
    files.sort()
    for fp in files:
        w('- %s (%.1f KB)' % (os.path.relpath(fp, pdir),
                              os.path.getsize(fp) / 1024))
    if files:
        w('\n### Representative payload — %s\n' % os.path.relpath(files[0], pdir))
        with open(files[0]) as f:
            payload = f.read()
        w('```json\n%s\n```' % payload[:12000])

    try:
        brain.close()
    except Exception:
        pass
    shutil.rmtree(work, ignore_errors=True)

    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, 'pooled_review_%s.md' % corpus_hash)
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('review → %s (%d lines)' % (path, len(out)))
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    _load_env()
    render(p.parse_args().corpus)
