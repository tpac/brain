#!/usr/bin/env python3
"""Consolidation Contract Eval — the dimensions-eval step of the S-scale
prompt-change process (prompt → examples → probe ↔ DIMENSIONS EVAL → platform A/B).

Mirrors eval/agent_introspect/encoder_contract_eval.py, for S2 consolidation.
An independent Opus evaluator scores each prompt arm's DECISIONS (the brain_batch
ops it emits per cluster, capture-only) against the 10-dim
consolidation_quality_contract — satisfied / degraded / violated / n_a — then
aggregates into a baseline-vs-candidate per-dimension table and applies the gate.

Independent scorer (Opus, not the Sonnet that produced the decisions) is
deliberate — kills the self-eval bias (brain node bc46e2fe).

Two modes (like encoder_contract_eval):
  - default: live A/B — run both arms over real clusters, score each decision.
  - --examples: score the candidate prompt's WORKED EXAMPLES as decisions
    (feed-examples-to-dimensions), so authoring blind spots surface before A/B.

Usage:
    ./dev python3 eval/agent_introspect/consolidation_contract_eval.py
    ./dev python3 eval/agent_introspect/consolidation_contract_eval.py --clusters 6 --save eval/reports/consol_dims.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.agent_introspect.consolidation_quality_contract import (
    DIMENSIONS, HARD_GATE_DIMS, AB_IMPROVEMENT_DIMS, VERDICTS, SCOPE)

CANDIDATE_PATH = os.path.join(
    ROOT, 'eval', 'candidate_prompts', 's2_consolidation_absorb.md')
INTERACTION = 's2_consolidation_enrichment'

_SCORE = {'satisfied': 1.0, 'degraded': 0.5, 'violated': 0.0}  # n_a excluded


EVALUATOR_SYSTEM_PROMPT = """You evaluate ONE S2 consolidation DECISION against a 10-dimension quality contract.

The S2 consolidation encoder is an autonomous Sonnet that merges duplicate memory
nodes while the operator is idle. For a cluster of similar nodes it chooses an
action and emits brain_batch ops:
- ABSORB (consolidate/evolve): `{op:"absorb", survivor_id, absorbed_id, content?, prune_edges?}`
  — folds the absorbed node INTO the survivor (source_refs/edges/access/metadata
  transfer automatically), then ARCHIVES the absorbed node. CRUCIAL: absorb keeps
  the SURVIVOR's content; it does NOT merge the absorbed node's content — so a
  `content` override is the ONLY way the absorbed node's unique content survives.
- KEEP / SKIP: `{op:"connect", relation:"similar_to", description}` — both nodes retained.

You receive:
- `cluster`: the member nodes (id, type, title, locked, content excerpt) + similarity cosines + decoder pre_class
- `decision`: the ops this arm emitted for THIS cluster
- `contract`: the 10 DIMENSIONS with satisfies / violates / degrades criteria

# Task
MECHANISM-BLIND — read this first. Two prompt arms are compared: an OLD arm that
merges via revise(survivor)+connect(edges)+archive(peer), and a NEW arm that
merges via a single `absorb` op. A correct decision scores the SAME whichever
ops achieved it. NEVER mark a dimension `violated` merely because an arm used
revise+connect+archive instead of absorb (or vice-versa). Judge the DECISION.

Each dimension carries a `scope` controlling when it applies — honor it strictly:
- scope=agnostic — always score (the decision's correctness, op-blind).
- scope=merge — score ONLY if the decision MERGES two nodes (via absorb, OR via
  revise-survivor+archive-peer). For a KEEP/SKIP decision → n_a. Both mechanisms
  must preserve the merged-away node's content (C4), leave an (id:) provenance
  trail in the survivor's content (C5), and pick the right survivor (C6).
- scope=keep — score ONLY for a KEEP/SKIP (similar_to) decision; for a merge → n_a.
- scope=absorb — score ONLY if the new `absorb` op was used; otherwise n_a.

Score EACH dimension C1..C10 independently:
- satisfied: positive signals present, no violations
- degraded: partial — some good, some degrade-list features visible
- violated: clear violation of the dim's criteria
- n_a: dimension's scope does not apply to this decision shape

Cite specific evidence (node types, titles, whether content/(id:) present, which
ids moved where). Don't hedge. Score on what the ops ACTUALLY do, not intent.

# Output STRICT JSON, no prose outside it:
{
  "per_dim": [
    {"dim": "C1_action_fit", "status": "satisfied|degraded|violated|n_a", "evidence": "<specific>"},
    ... all 10 ...
  ],
  "summary": "<one line: did this decision merge losslessly without over-merging or churn?>"
}"""


import re


def validate_example_authoring(prompt_text):
    """Mechanical authoring gate on the prompt's worked examples (no LLM).

    Enforces the discipline that the examples ARE the training signal:
      1. No truncated content — examples must be full text (no '...' / '…').
      2. Every `absorb` op authors BOTH `title:` and `content:` (mandatory rewrite;
         a content-less or title-stale absorb is the bug we're preventing).
      3. The example set collectively exemplifies EVERY caller capability the
         encoder is expected to use — so each is demonstrated, not just described.

    Returns {ok: bool, failures: [...], coverage: {...}}.
    """
    blocks = re.findall(r'```(.*?)```', prompt_text, re.DOTALL)
    # Anti-pattern (BAD) example blocks are intentionally wrong — exclude them
    # from the gate (they're marked with ❌ / ANTI-PATTERN). Only GOOD examples
    # are the training signal we validate.
    good = [b for b in blocks if '❌' not in b and 'ANTI-PATTERN' not in b]
    code = '\n'.join(good)
    failures = []

    # 1. no truncation inside example code
    for marker in ('...', '…'):
        if marker in code:
            # show a little context per occurrence
            idx = code.find(marker)
            failures.append("truncation marker %r in example code near: ...%s..."
                            % (marker, code[max(0, idx-40):idx+10].replace('\n', ' ')))

    # 2. split into ops by the flat `{op:` boundary (ops don't nest here)
    chunks = re.split(r'\{op:\s*', code)[1:]
    ops = []
    for ch in chunks:
        m = re.match(r'"(\w+)"', ch)
        if not m:
            continue
        op = m.group(1)
        body = ch.split('{op:')[0]   # up to the next op (defensive; split already did this)
        ops.append((op, body))
        if op == 'absorb':
            # content is MANDATORY on every absorb (the losslessness guard).
            ok_content = bool(re.search(r'\bcontent:\s*"[^"]{20,}', body))
            if not ok_content:
                failures.append("absorb op missing/short content (content is mandatory): %s"
                                % body[:60].strip())
            # title is CONDITIONAL — rewritten when the merge changes the subject,
            # kept (omitted) for a title-stable append. Tracked for coverage below,
            # not hard-failed per-op.

    # 3. capability coverage across the whole example set
    opnames = {o for o, _ in ops}
    coverage = {
        'absorb_title_content': any(
            o == 'absorb' and re.search(r'title:\s*"', b) and re.search(r'content:\s*"', b)
            for o, b in ops),
        # both merge patterns must be demonstrated:
        'title_rewrite_pattern': any(
            o == 'absorb' and re.search(r'title:\s*"', b) for o, b in ops),
        'title_keep_append_pattern': any(
            o == 'absorb' and re.search(r'content:\s*"', b) and not re.search(r'title:\s*"', b)
            for o, b in ops),
        'kv_revise_on_absorb': any(
            o == 'absorb' and re.search(r'\b(situation|keywords|reasoning):\s*"', b)
            for o, b in ops),
        'drop_fields': 'drop_fields' in code,
        'prune_edges': 'prune_edges' in code,
        'connect_similar_to': bool(re.search(r'relation:\s*"similar_to"', code)),
        'connect_correction': bool(re.search(r'relation:\s*"(corrects|supersedes|reframes|resolves)"', code)),
        'revise_op': 'revise' in opnames,
        'disconnect_op': 'disconnect' in opnames,
        'connect_op': 'connect' in opnames,
    }
    for cap, present in coverage.items():
        if not present:
            failures.append("capability NOT exemplified: %s" % cap)

    return {'ok': not failures, 'failures': failures, 'coverage': coverage,
            'n_ops': len(ops), 'n_absorb': sum(1 for o, _ in ops if o == 'absorb')}


def load_contract_summary() -> str:
    lines = [f"# {len(DIMENSIONS)}-DIMENSION CONSOLIDATION QUALITY CONTRACT", ""]
    for name, dim in DIMENSIONS.items():
        lines.append(f"## {name}")
        lines.append(f"**Scope**: {SCOPE.get(name, 'agnostic')}"
                     + ("  [HARD GATE]" if dim.get('hard_gate') else ""))
        lines.append(f"**Group**: {dim['group']}")
        lines.append(f"**Intent**: {dim['intent']}")
        lines.append("**Satisfies**:")
        lines += [f"  - {s}" for s in dim.get('satisfies', [])]
        lines.append("**Violates**:")
        lines += [f"  - {v}" for v in dim.get('violates', [])]
        if dim.get('degrades'):
            lines.append("**Degrades**:")
            lines += [f"  - {d}" for d in dim['degrades']]
        lines.append("")
    return "\n".join(lines)


def render_cluster(brain, cluster):
    ids = cluster.get('nodes', [])
    rows = {}
    if ids:
        ph = ','.join('?' * len(ids))
        for r in brain.conn.execute(
                "SELECT id, type, title, content, locked, critical "
                "FROM nodes WHERE id IN (%s)" % ph, ids):
            rows[r[0]] = {'id': r[0][:8], 'type': r[1], 'title': (r[2] or '')[:80],
                          'content_excerpt': (r[3] or '')[:300],
                          'locked': bool(r[4]) or bool(r[5])}
    return {
        'pre_class': cluster.get('pre_class'),
        'content_cosine': round(cluster.get('content_cosine_max', 0), 3),
        'title_cosine': round(cluster.get('title_cosine_max', 0), 3),
        'members': [rows.get(nid, {'id': nid[:8], 'type': '?'}) for nid in ids],
    }


def ops_for_cluster(ops, cluster_ids):
    s = set(cluster_ids)
    out = []
    for o in ops:
        touched = {o.get('survivor_id'), o.get('absorbed_id'), o.get('node_id'),
                   o.get('source_id'), o.get('target_id')}
        if touched & s:
            # compact, score-relevant view
            view = {'op': o.get('op')}
            for k in ('survivor_id', 'absorbed_id', 'node_id', 'source_id',
                      'target_id', 'relation'):
                if o.get(k):
                    view[k] = o[k][:8] if k.endswith('_id') else o[k]
            if o.get('op') == 'absorb':
                view['has_content'] = bool((o.get('content') or '').strip())
                view['has_id_ref'] = '(id:' in (o.get('content') or '')
                view['content_excerpt'] = (o.get('content') or '')[:300]
                if o.get('prune_edges'):
                    view['prune_edges'] = o['prune_edges']
            if o.get('description'):
                view['description'] = o['description'][:200]
            out.append(view)
    return out


def score_decision(contract_summary, cluster_render, decision_ops):
    from eval.agent_introspect._common import call_sonnet, OPUS_MODEL
    payload = {'cluster': cluster_render, 'decision': decision_ops or '(no op emitted)'}
    msg = ("contract:\n" + contract_summary
           + "\n\n=== DECISION TO SCORE ===\n" + json.dumps(payload, indent=2))
    # NOTE: opus-4-8 deprecates `temperature` (400 if passed) — newer models drop
    # the knob. Grader runs at the model's default (already low-variance); the
    # K-sample loop is what handles variance, not a pinned grader temp.
    out = call_sonnet(EVALUATOR_SYSTEM_PROMPT, msg, max_tokens=2048, model=OPUS_MODEL)
    text = out['text']
    # tolerate ```json fences
    t = text.strip()
    if t.startswith('```'):
        t = t.split('```', 2)[1].lstrip('json').strip() if '```' in t else t
    try:
        return json.loads(t)
    except Exception:
        return {'per_dim': [], 'summary': 'PARSE_ERROR', '_raw': text[:500]}


def aggregate(arm_results):
    """arm_results: list of per-cluster score dicts. Returns per-dim {status: count}
    and a per-dim mean score (n_a excluded)."""
    counts = {d: defaultdict(int) for d in DIMENSIONS}
    scores = {d: [] for d in DIMENSIONS}
    for res in arm_results:
        for entry in res.get('per_dim', []):
            d, st = entry.get('dim'), entry.get('status')
            if d not in DIMENSIONS or st not in VERDICTS:
                continue
            counts[d][st] += 1
            if st in _SCORE:
                scores[d].append(_SCORE[st])
    means = {d: (sum(v) / len(v) if v else None) for d, v in scores.items()}
    return counts, means


def gate(base_means, cand_means, cand_counts):
    """Apply the contract gate. Returns (passed, reasons)."""
    reasons = []
    # HARD GATE — zero violated on hard dims (candidate)
    for d in HARD_GATE_DIMS:
        v = cand_counts[d].get('violated', 0)
        if v:
            reasons.append('HARD FAIL %s: %d violated' % (d, v))
    # A/B — candidate >= baseline everywhere, strictly > on improvement dims
    eps = 1e-9
    for d in DIMENSIONS:
        b, c = base_means.get(d), cand_means.get(d)
        if b is None or c is None:
            continue
        if c < b - eps:
            reasons.append('REGRESSION %s: candidate %.2f < baseline %.2f' % (d, c, b))
    for d in AB_IMPROVEMENT_DIMS:
        b, c = base_means.get(d), cand_means.get(d)
        if b is None or c is None:
            continue
        # require improvement — UNLESS both are already at the ceiling (you can't
        # beat a perfect baseline; a tie at 1.0 is not a regression).
        at_ceiling = c >= 1.0 - eps and b >= 1.0 - eps
        if not (c > b + eps) and not at_ceiling:
            reasons.append('NO GAIN %s: candidate %.2f !> baseline %.2f (must improve)' % (d, c, b))
    return (len(reasons) == 0), reasons


def _fmt_means(label, means):
    cells = []
    for d in DIMENSIONS:
        m = means.get(d)
        cells.append('%s=%s' % (d.split('_')[0], '—' if m is None else '%.2f' % m))
    return '%-10s %s' % (label, '  '.join(cells))


def main():
    ap = argparse.ArgumentParser(description='S2 consolidation contract eval')
    ap.add_argument('--clusters', type=int, default=6)
    ap.add_argument('--samples', type=int, default=1,
                    help='Encoder samples per arm (variance analysis). The '
                         'encoder is non-deterministic even at temp 0, so n=1 '
                         'is unreliable — use 3-5 for a gating verdict.')
    ap.add_argument('--save')
    ap.add_argument('--keep', action='store_true')
    ap.add_argument('--validate', action='store_true',
                    help='Mechanical authoring gate on the candidate examples (no LLM, no brain)')
    args = ap.parse_args()

    if args.validate:
        r = validate_example_authoring(open(CANDIDATE_PATH).read())
        print('authoring gate: %s  (%d ops, %d absorb)' % (
            'PASS' if r['ok'] else 'FAIL', r['n_ops'], r['n_absorb']))
        for cap, ok in r['coverage'].items():
            print('  %s %s' % ('✓' if ok else '✗', cap))
        for f in r['failures']:
            print('  FAIL: ' + f)
        return

    from tests.isolated_brain import IsolatedBrain
    from eval.s2_consolidation_eval import run_decoder, run_capture_variant
    from eval.agent_introspect._common import load_env

    contract_summary = load_contract_summary()
    with open(CANDIDATE_PATH) as f:
        candidate = f.read().strip()

    load_env()
    print('Setting up isolated brain copy (production snapshot)...')
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        baseline = brain.get_interaction_prompt(INTERACTION)

        decode = run_decoder(brain)
        clusters = decode.get('clusters', [])[:args.clusters]
        print('Scoring %d clusters × %d samples × 2 arms on %d dims '
              '(encoder temp 0, grader Opus temp 0)...' % (
                  len(clusters), args.samples, len(DIMENSIONS)))

        arms = {}
        for label, prompt in (('baseline', baseline), ('candidate', candidate)):
            results = []
            for s in range(args.samples):
                variant = run_capture_variant(brain, clusters, prompt)
                for c in clusters:
                    cr = render_cluster(brain, c)
                    ops = ops_for_cluster(variant['ops'], c.get('nodes', []))
                    results.append(score_decision(contract_summary, cr, ops))
            arms[label] = results

        base_counts, base_means = aggregate(arms['baseline'])
        cand_counts, cand_means = aggregate(arms['candidate'])
        passed, reasons = gate(base_means, cand_means, cand_counts)

        print('\n=== PER-DIMENSION MEANS (1.0=satisfied, 0.5=degraded, 0=violated, —=n/a) ===')
        print(_fmt_means('baseline', base_means))
        print(_fmt_means('candidate', cand_means))
        print('\n=== HARD-GATE DIM VIOLATIONS (candidate) ===')
        for d in HARD_GATE_DIMS:
            print('  %-26s violated=%d  degraded=%d  satisfied=%d  n_a=%d' % (
                d, cand_counts[d].get('violated', 0), cand_counts[d].get('degraded', 0),
                cand_counts[d].get('satisfied', 0), cand_counts[d].get('n_a', 0)))
        print('\n=== GATE: %s ===' % ('PASS' if passed else 'FAIL'))
        for r in reasons:
            print('  - ' + r)

        if args.save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            with open(args.save, 'w') as f:
                json.dump({
                    'passed': passed, 'reasons': reasons,
                    'baseline_means': base_means, 'candidate_means': cand_means,
                    'baseline_counts': {d: dict(c) for d, c in base_counts.items()},
                    'candidate_counts': {d: dict(c) for d, c in cand_counts.items()},
                    'baseline_results': arms['baseline'],
                    'candidate_results': arms['candidate'],
                }, f, indent=2, default=str)
            print('\nSaved report to %s' % args.save)


if __name__ == '__main__':
    main()
