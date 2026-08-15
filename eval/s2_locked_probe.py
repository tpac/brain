#!/usr/bin/env python3
"""S2 locked-absorb PROBE — empirical reproduction of the real failing instance.

The real failure: cluster {96d2fdf8, 426ae3cd} — two operator-LOCKED, near-identical
"interactions table" architecture nodes that already share a `similar_to` AND a
`consolidated_into` edge. The consolidation encoder keeps "CONSOLIDATE"-revising
the survivor every idle cycle (churn) and intermittently tries to archive the
locked peer (archive_guarded false-error, 14x observed).

This probe drives that EXACT cluster through the consolidation encoder under:
  A. baseline  — the current ACTIVE prompt (s2_consolidation_enrichment v5)
  B. candidate — v5 + the locked-absorb hard-rules reframe (surgical string edits)
  C. reasoning — an open differential question, to learn what Sonnet thinks is correct

Safety:
  - Runs against an IsolatedBrain copy of production. Live data is never touched.
  - Dispatch is CAPTURE-ONLY: brain_batch ops are recorded, not applied — so A and B
    see byte-identical input (no state divergence) and we observe the DECISION, not
    the mutation. (Lossless-merge verification is a separate, real-dispatch phase.)

Usage:
    ./dev python3 eval/s2_locked_probe.py
    ./dev python3 eval/s2_locked_probe.py --save eval/reports/locked_probe.json
    ./dev python3 eval/s2_locked_probe.py --no-reasoning   # skip Probe C
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TARGET = '96d2fdf8'                 # the locked node the encoder kept trying to archive
PAIR = ('426ae3cd', '96d2fdf8')    # the double-locked pair (sorted)


# ──────────────────────────────────────────────────────────────────
# Candidate prompt — surgical edits to the ACTIVE v5 (locked-absorb reframe)
# ──────────────────────────────────────────────────────────────────

HARD_RULES = """## Locked Nodes — Hard Rules (override every other signal)

Locked / critical nodes are operator-sacred. These rules OVERRIDE every signal
below — CATALOG_BLIND, CORRECTION_EDGE, high cosine, all of it.

1. **A locked node is NEVER archived.** Not as a peer, not "after enhancing the
   survivor," never. Do not emit an archive op for a locked/critical node.

2. **Two or more locked nodes in a cluster → KEEP, never ABSORB.** Do NOT revise
   one locked node to absorb another. That is not consolidation — it is churn:
   the revise re-touches the node every cycle, the cluster re-proposes forever,
   and you redo the same merge endlessly. Two sacred nodes cannot be collapsed.
   - **KEEP via a single `similar_to` edge, then STOP.** A `connect` is idempotent
     and does NOT touch node timestamps, so re-asserting an edge is harmless — but
     a `revise` IS the churn bug: revising a locked node bumps its timestamp and
     re-arms the cluster every cycle. So emit at most the `similar_to` edge, never
     a revise. If you can already see a `similar_to` edge between them in the
     data, emit nothing at all.
   - **When the two locked nodes are true duplicates** (same knowledge, not just
     adjacent), say so in the edge description and flag it for the operator —
     e.g. "duplicate — merge candidate, requires operator to unlock one." The
     edge becomes the actionable record: only the operator can unlock, and once
     one is unlocked a later run absorbs it cleanly.

3. **One locked + unlocked redundant neighbor(s) → the locked node is the survivor
   AND the absorb-target.** Revise the LOCKED node to absorb the unlocked
   neighbor's unique detail (locked blocks unlocking/deletion, NOT content
   enrichment), migrate edges into it, then archive the UNLOCKED peer. Knowledge
   flows toward the canonical node.

4. **Contradiction is not redundancy.** If an unlocked neighbor contradicts or
   supersedes a locked node, do NOT absorb it and do NOT archive the locked node.
   Add a `corrects`/`supersedes` edge (unlocked → locked) so the tension is
   recorded, and leave the locked node's content intact. Consolidation handles
   duplication, not corrections.

"""


def build_candidate(baseline: str) -> str:
    """Produce the candidate prompt via explicit, asserted string edits.

    Every replacement asserts it changed something so a silent prompt-drift
    no-match can never pass as a valid candidate.
    """
    edits = [
        # 1. Survivor bullet → point at the hard rules + ladder framing
        ("- **`locked` or `critical`** — must survive. Operator-curated; sacred. "
         "If both are locked, you can't ABSORB — do KEEP.",
         "- **`locked` or `critical`** — see **Locked Nodes — Hard Rules** below. "
         "Locked is the top of the canonicity ladder: always the survivor / "
         "absorb-target, never archived; two locked → KEEP."),
        # 2. Evidence line → hard rules override the ABSORB signals
        ("- **LOCKED / CRITICAL** — must survive. If both locked → KEEP.",
         "- **LOCKED / CRITICAL** — governed by **Locked Nodes — Hard Rules**, "
         "which OVERRIDE every signal above (CATALOG_BLIND and CORRECTION_EDGE "
         "included). A locked node is never archived; two locked → KEEP, and if "
         "a `similar_to` edge already exists → emit nothing."),
        # 3. Carve out the settled-locked-pair exception
        ("Process ALL clusters in the batch. Each results in ABSORB (revise + "
         "archive covers suppression) or a `similar_to` edge (KEEP/SKIP). "
         "Skipping a cluster without either leaves it in the backlog.",
         "Process ALL clusters in the batch. Each results in ABSORB (revise + "
         "archive covers suppression) or a `similar_to` edge (KEEP/SKIP). "
         "Skipping a cluster without either leaves it in the backlog. "
         "**EXCEPTION:** a cluster of ≥2 locked nodes that already share a "
         "`similar_to` edge is already settled — emit nothing for it (see "
         "Locked Nodes — Hard Rules)."),
    ]
    cand = baseline
    for old, new in edits:
        if old not in cand:
            raise SystemExit("CANDIDATE BUILD FAILED: anchor not found in v5:\n  %s..."
                             % old[:80])
        cand = cand.replace(old, new, 1)

    # 4. Insert the hard-rules section just before "## Actions"
    if "## Actions\n" not in cand:
        raise SystemExit("CANDIDATE BUILD FAILED: '## Actions' header not found")
    cand = cand.replace("## Actions\n", HARD_RULES + "## Actions\n", 1)
    return cand


# ──────────────────────────────────────────────────────────────────
# Cluster acquisition
# ──────────────────────────────────────────────────────────────────

def get_target_cluster(brain):
    """Find the real {96d2fdf8, 426ae3cd} cluster via the decoder; hand-build
    via the real enrichment path if the cold-start scan doesn't surface it."""
    from eval.s2_consolidation_eval import run_decoder
    decode = run_decoder(brain)
    for c in decode.get('clusters', []):
        if any(n == TARGET for n in c['nodes']):
            return c, 'decoder-scan', decode['stats']

    # Fallback: hand-build the pair, enrich via the REAL decoder enrichment
    from servers.scales.s2.consolidation_decoder import ConsolidationDecoder
    d = ConsolidationDecoder(brain)
    key = '%s-%s' % PAIR
    base = [{
        'nodes': list(PAIR), 'size': 2,
        'content_cosine_max': 0.90, 'content_cosine_avg': 0.90,
        'title_cosine_max': 0.80, 'title_cosine_avg': 0.80,
        'pair_scores': {key: {'content': 0.90, 'title': 0.80}},
    }]
    enriched = d._enrich_clusters(base)
    if not enriched:
        raise SystemExit("Could not build the target cluster — both nodes present?")
    enriched[0]['pre_class'] = d._pre_classify(enriched[0])
    return enriched[0], 'hand-built', decode.get('stats', {})


# ──────────────────────────────────────────────────────────────────
# Variant runner — capture-only dispatch, prompt swap, journal write disabled
# ──────────────────────────────────────────────────────────────────

def run_variant(brain, cluster, prompt_text, label):
    from servers.daemon_dispatch import COMMAND_TABLE
    from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
    from servers.scales.s2.consolidation_contract import CONSOLIDATION

    captured = []

    def dispatch(cmd, cmd_args):
        if cmd == 'brain_batch':
            captured.append(cmd_args)
            ops = cmd_args.get('operations', []) if isinstance(cmd_args, dict) else []
            return {'ok': True, 'result': {'dry_run': True, 'ops_seen': len(ops)}}
        # get_nodes and any read → real handler against the isolated copy
        entry = COMMAND_TABLE.get(cmd)
        if entry:
            return entry.handler(brain, cmd_args, [])
        return {'ok': True, 'result': {}}

    orig_prompt = brain.get_interaction_prompt
    brain.get_interaction_prompt = (
        lambda name: prompt_text if name == 's2_consolidation_enrichment'
        else orig_prompt(name))
    try:
        enc = ConsolidationEncoder(brain, dispatch_fn=dispatch, config=CONSOLIDATION)
        enc._save_journal = lambda *a, **k: ''   # don't let run A contaminate run B
        result = enc.run([cluster]) or {}
    finally:
        brain.get_interaction_prompt = orig_prompt

    # Flatten captured brain_batch ops
    ops = []
    for cmd_args in captured:
        for op in (cmd_args.get('operations', []) if isinstance(cmd_args, dict) else []):
            if isinstance(op, dict):
                ops.append(op)
    return {
        'label': label,
        'ops': ops,
        'rounds': result.get('rounds', 0),
        'final_text': result.get('final_text', '') or '',
    }


def analyze(variant, locked_ids):
    ops = variant['ops']
    archives = [o for o in ops if o.get('op') == 'archive']
    revises = [o for o in ops if o.get('op') == 'revise']
    connects = [o for o in ops if o.get('op') == 'connect']

    def _nid(o):
        return o.get('node_id') or o.get('id') or ''

    locked_archive_attempts = [o for o in archives if _nid(o) in locked_ids]
    locked_revises = [o for o in revises if _nid(o) in locked_ids]
    similar_to = [o for o in connects if o.get('relation') == 'similar_to']

    return {
        'n_ops': len(ops),
        'archives': [_nid(o) for o in archives],
        'revises': [_nid(o) for o in revises],
        'connects': [(o.get('source_id', '')[:8], o.get('target_id', '')[:8],
                      o.get('relation', '')) for o in connects],
        'locked_archive_attempts': [_nid(o) for o in locked_archive_attempts],
        'locked_revises': [_nid(o) for o in locked_revises],
        'similar_to_emitted': len(similar_to),
        'emitted_nothing': len(ops) == 0,
    }


# ──────────────────────────────────────────────────────────────────
# Probe C — open differential reasoning question (no tools)
# ──────────────────────────────────────────────────────────────────

REASONING_PROMPT = """You are reviewing an S2 consolidation decision for a persistent memory graph.

A cluster contains TWO nodes, BOTH operator-locked (sacred, never archivable),
same type, near-identical content about the same architecture concept. They
ALREADY have a `similar_to` edge AND a `consolidated_into` edge between them. The
decoder keeps re-proposing this exact pair every idle cycle.

Answer concisely:
1. What is the single correct consolidation action here, and why?
2. A prior encoder run repeatedly chose to "revise node B to absorb node A's
   content" (no archive, since A is locked). What is wrong with that for the
   graph's long-term behavior?
3. One option is to leave both untouched — emit NO operation. When is that the
   right call, and when would it be wrong?"""


def run_reasoning_probe(brain):
    import anthropic
    from servers.brain_constants import ANTHROPIC_CLIENT_TIMEOUT
    from servers.scales.dispatch import load_env
    if not os.environ.get('ANTHROPIC_API_KEY'):
        load_env()
    client = anthropic.Anthropic(timeout=ANTHROPIC_CLIENT_TIMEOUT)
    resp = client.messages.create(
        model='claude-sonnet-4-6', max_tokens=1024,
        messages=[{'role': 'user', 'content': REASONING_PROMPT}])
    return ''.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')


# ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='S2 locked-absorb probe')
    ap.add_argument('--save', help='Write full JSON report here')
    ap.add_argument('--no-reasoning', action='store_true', help='Skip Probe C')
    ap.add_argument('--keep', action='store_true', help='Keep isolated copy')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain

    print("Setting up isolated brain copy (production snapshot)...")
    with IsolatedBrain(cleanup=not args.keep) as env:
        brain = env.brain
        if args.keep:
            print("Isolated brain at:", env.db_dir)

        # Confirm both nodes exist and are locked in the copy
        rows = brain.conn.execute(
            "SELECT id, locked, title FROM nodes WHERE id IN (?, ?)", PAIR).fetchall()
        locked_ids = {r[0] for r in rows if r[1]}
        print("Pair state in copy:")
        for r in rows:
            print("  %s locked=%s  %s" % (r[0], bool(r[1]), (r[2] or '')[:50]))
        if len(rows) != 2:
            raise SystemExit("Both nodes must exist in the copy — found %d" % len(rows))

        baseline = brain.get_interaction_prompt('s2_consolidation_enrichment')
        candidate = build_candidate(baseline)
        print("\nCandidate prompt built: %d → %d chars (+%d)" % (
            len(baseline), len(candidate), len(candidate) - len(baseline)))

        print("\nAcquiring the real cluster...")
        cluster, source, _stats = get_target_cluster(brain)
        print("  cluster source=%s  nodes=%s  pre_class=%s  content_cosine=%.3f" % (
            source, cluster['nodes'], cluster.get('pre_class'),
            cluster.get('content_cosine_max', 0)))

        print("\n=== PROBE A — baseline (ACTIVE v5) ===")
        a = run_variant(brain, cluster, baseline, 'baseline-v5')
        ra = analyze(a, locked_ids)
        print_variant(a, ra)

        print("\n=== PROBE B — candidate (locked-absorb reframe) ===")
        b = run_variant(brain, cluster, candidate, 'candidate')
        rb = analyze(b, locked_ids)
        print_variant(b, rb)

        reasoning = None
        if not args.no_reasoning:
            print("\n=== PROBE C — differential reasoning ===")
            reasoning = run_reasoning_probe(brain)
            print(reasoning)

        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)
        print("                          baseline-v5      candidate")
        print("  ops emitted             %-16d %d" % (ra['n_ops'], rb['n_ops']))
        print("  locked archive attempt  %-16s %s" % (
            bool(ra['locked_archive_attempts']), bool(rb['locked_archive_attempts'])))
        print("  locked node revised     %-16s %s" % (
            bool(ra['locked_revises']), bool(rb['locked_revises'])))
        print("  emitted nothing         %-16s %s" % (
            ra['emitted_nothing'], rb['emitted_nothing']))
        print("\n  baseline reproduces churn/false-error: %s" % (
            bool(ra['locked_archive_attempts']) or bool(ra['locked_revises'])))
        print("  candidate avoids both:                 %s" % (
            not ra_bad(rb)))

        if args.save:
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            with open(args.save, 'w') as f:
                json.dump({
                    'cluster_source': source,
                    'cluster_nodes': cluster['nodes'],
                    'pre_class': cluster.get('pre_class'),
                    'baseline': {'analysis': ra, 'final_text': a['final_text']},
                    'candidate': {'analysis': rb, 'final_text': b['final_text']},
                    'reasoning': reasoning,
                }, f, indent=2, default=str)
            print("\nSaved report to %s" % args.save)


def ra_bad(r):
    return bool(r['locked_archive_attempts']) or bool(r['locked_revises'])


def print_variant(v, r):
    print("  rounds=%d  ops=%d" % (v['rounds'], r['n_ops']))
    for o in v['ops']:
        nid = o.get('node_id') or o.get('id') or o.get('source_id', '')
        extra = ''
        if o.get('op') == 'connect':
            extra = ' %s→%s (%s)' % (o.get('source_id', '')[:8],
                                          o.get('target_id', '')[:8],
                                          o.get('relation', ''))
        print("    - %-9s %s%s" % (o.get('op', '?'), str(nid)[:8], extra))
    if r['locked_archive_attempts']:
        print("    ⚠ ARCHIVE attempt on LOCKED: %s" % r['locked_archive_attempts'])
    if r['locked_revises']:
        print("    ⚠ REVISE on LOCKED (churn): %s" % r['locked_revises'])
    if r['emitted_nothing']:
        print("    ✓ emitted nothing (settled)")
    # Journal tail
    jt = v['final_text'].strip()
    if jt:
        print("    journal: %s" % jt.replace('\n', ' ')[:200])


if __name__ == '__main__':
    main()
