#!/usr/bin/env python3
"""Run a REAL consolidation cycle on an IsolatedBrain copy with the v9 prompt +
the unified prompt closers (edge-aspects → review block → closure), and observe:

  1. Assembly — the assembled system prompt has `## Edge Aspects` (renamed, no
     survivor_lineage / generic_relation / noise), the review block, the closure
     LAST ending in "DONE", and NO dead `## Encoding Journal` / `Round 2` /
     continuity bullet.
  2. Accept + decline mix — the fed batch spans likely_consolidate / likely_evolve
     (absorb-leaning) AND needs_judgment (keep/skip-leaning), and the encoder both
     absorbs (writes) and connects (similar_to). Tom's "both reject and accept".
  3. Residue — encoder emits `## Review`; notes parse clean; persisted == parsed;
     malformed == 0 (no regression from the rewrite).

Never touches live data (IsolatedBrain copies brain.db + brain_logs.db to a temp
dir, cleaned on exit). v9 is derived HERE from the live v8 via make_v9() — the
exact transform reused at landing — and activated on the isolated brain.

    ./dev python3 eval/sim_consolidation_journal.py [n_clusters]
"""
import os
import re
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.isolated_brain import IsolatedBrain
from tests.interaction_override import override_interaction
from servers.trace_contract import extract_review_block, parse_journal_notes
from servers.scales.s2.consolidation_encoder import ConsolidationEncoder
from servers.scales.s2.consolidation_contract import CONSOLIDATION
from eval.s2_consolidation_eval import run_decoder

N_CLUSTERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def make_v9(v8: str) -> str:
    """v8 → v9: drop the continuity bullet, reword `## Speed` (no rounds /
    journal / DONE), delete the dead `## Encoding Journal` section. Loud asserts
    catch prompt drift — reused verbatim at landing."""
    bullet = "- **CONSOLIDATION JOURNAL** — what previous runs decided. Your continuity.\n"
    assert v8.count(bullet) == 1, 'continuity-bullet anchor not unique (%d)' % v8.count(bullet)
    out = v8.replace(bullet, "")

    old_speed = ("## Speed\n\n"
                 "Target: **2 rounds.**\n"
                 "- Round 1: read clusters. If you need a deeper look at any node, "
                 "call `get_nodes`. Then `brain_batch` with all actions.\n"
                 "- Round 2: journal + DONE.\n\n"
                 "Do NOT recall or search — everything you need is in the cluster data.")
    new_speed = ("## Speed\n\n"
                 "Be decisive. One optional `get_nodes` if you need a deeper look, "
                 "then `brain_batch` with all actions — don't over-inspect.\n\n"
                 "Do NOT recall or search — everything you need is in the cluster data.")
    assert out.count(old_speed) == 1, 'Speed anchor not unique (%d)' % out.count(old_speed)
    out = out.replace(old_speed, new_speed)

    j = out.find("## Encoding Journal")
    assert j != -1, 'Encoding Journal section missing'
    return out[:j].rstrip()


def _parse_per_batch(final_text):
    """Re-parse the encoder's concatenated multi-batch final_text the way the
    per-batch write_journal_notes did. Returns (well_formed, malformed, sections)."""
    chunks = re.split(r'\n--- batch \d+ ---\n', final_text)
    wf, mal, sections = 0, 0, 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        block = extract_review_block(chunk)
        if block is None:           # no `## Review` / broken fence — not a section
            continue
        sections += 1               # block == '' (clean empty) or fence content
        notes, m = parse_journal_notes(block)
        wf += len(notes)
        mal += len(m)
    return wf, mal, sections


def main():
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        print('isolated brain: %d non-archived nodes' % env.node_count())

        # 1. Derive v9 from the live v8 and activate it on the isolated brain.
        v8 = brain.get_interaction_prompt('s2_consolidation_enrichment') or ''
        assert v8, 'no s2_consolidation_enrichment prompt in isolated brain'
        v9 = make_v9(v8)
        params = brain.get_interaction_config('s2_consolidation_enrichment') or {}
        override_interaction(brain, 's2_consolidation_enrichment', template=v9,
                             parameters=params, set_by='eval:journal_v9')

        # 1b. Assembly check (deterministic) — exactly what the encoder will build.
        enc0 = ConsolidationEncoder(brain, config=CONSOLIDATION)
        asm = enc0.journal.decorate_system(enc0._inject_edge_aspects(v9))
        checks = {
            'edge aspects renamed':        '## Edge Aspects' in asm and '## Edge Families' not in asm,
            'survivor_lineage excluded':   'absorbed_into' not in asm,
            'no dead Encoding Journal':    '## Encoding Journal' not in asm,
            'no Round 2 / continuity bullet': 'Round 2' not in asm and 'CONSOLIDATION JOURNAL' not in asm,
            'review block present':        'Put your notes under a `## Review`' in asm,
            'closure last, ends DONE':     asm.rstrip().endswith('"DONE".'),
            'order edge<review<closure':   asm.index('## Edge Aspects') < asm.index('Your review') < asm.index('## Finishing'),
        }
        print('\n=== assembly ===')
        for k, v in checks.items():
            print('  %s %s' % ('✓' if v else '✗', k))

        # 2. Decode (cold scan) → select a batch spanning absorb-leaning AND
        #    judgment classes, so both accept (absorb) and decline (KEEP/SKIP)
        #    branches are exercised on real production clusters.
        dec = run_decoder(brain)
        clusters = dec['clusters']
        print('\n=== decoder ===  clusters=%d  class_counts=%s'
              % (len(clusters), dec['stats'].get('class_counts')))
        if not clusters:
            print('no clusters — graph settled; nothing to encode this run.')
            return
        by_cls = {}
        for c in clusters:
            by_cls.setdefault(c.get('pre_class', '?'), []).append(c)
        per = max(1, N_CLUSTERS // 3)
        batch = []
        for cls in ('likely_consolidate', 'likely_evolve', 'needs_judgment'):
            batch += by_cls.get(cls, [])[:per]
        for c in clusters:           # top up if a class was thin
            if len(batch) >= N_CLUSTERS:
                break
            if c not in batch:
                batch.append(c)
        batch = batch[:N_CLUSTERS]
        print('  selected batch=%d  spans=%s'
              % (len(batch), dict(Counter(c.get('pre_class', '?') for c in batch))))

        # 3. REAL encode (one batch → one Sonnet call).
        cfg = dict(CONSOLIDATION)
        cfg['max_proposals_per_call'] = N_CLUSTERS
        encoder = ConsolidationEncoder(brain, config=cfg)
        print('\n--- encoder (REAL Sonnet run) ---')
        t0 = time.time()
        result = encoder.run(batch) or {}
        dt = time.time() - t0
        final_text = result.get('final_text', '') or ''
        print('actions=%d writes=%d rounds=%d  %.1fs'
              % (result.get('actions', 0), result.get('write_actions', 0),
                 result.get('rounds', 0), dt))

        # 3a. action mix — absorb (accept) vs connect (decline: KEEP/SKIP).
        ops = []
        for d in result.get('action_details', []):
            for op in ((d.get('input') or {}).get('operations') or []):
                ops.append(op.get('op'))
        print('=== action mix ===  ops=%s  archived=%d'
              % (dict(Counter(ops)), len(result.get('archived', []))))

        # 3b. residue — re-parse per batch, then authoritative read-back for THIS
        #     run's chain_id (IsolatedBrain copies live logs → filter, else prior
        #     runs' notes inflate the count).
        wf, mal, sections = _parse_per_batch(final_text)
        run_chain = encoder.chain_id()
        rows = [r for r in brain.journal_notes(scale='s2', unit='consolidation', k=1)
                if r.get('chain_id') == run_chain]
        print('\n=== residue ===  review-sections=%d  parsed=%d  MALFORMED=%d  persisted(this run)=%d'
              % (sections, wf, mal, len(rows)))
        for r in rows:
            print('  %s · %s · %s' % (
                (r.get('tag') or '—'), r.get('subject', ''), (r.get('note') or '')[:180]))

        # Verdict.
        print('\n=== verdict ===')
        assembly_ok = all(checks.values())
        residue_ok = (sections > 0 and mal == 0 and wf == len(rows))
        mix_ok = ('absorb' in ops) and ('connect' in ops)
        print('  %s assembly correct (edge-aspects renamed, closure last, no dead journal)'
              % ('✓' if assembly_ok else '✗'))
        print('  %s residue clean (sections>0, 0 malformed, parsed==persisted)'
              % ('✓' if residue_ok else '✗'))
        print('  %s accept + decline both exercised (absorb AND connect ops)'
              % ('✓' if mix_ok else '⚠'))


if __name__ == '__main__':
    main()
