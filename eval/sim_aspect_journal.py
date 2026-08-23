#!/usr/bin/env python3
"""Run a REAL aspect-integration cycle on an IsolatedBrain copy with the v8
prompt + the journal binding on _call_llm (Phase 3), and observe:

  1. Assembly — review block at the system tail, no closure, no arc; the
     JSON-only line softened so it can't suppress the `## Review` fence.
  2. Payload survival — Sonnet's classifications JSON parses after the
     harvest strip; validation accepts them; taxonomy write goes through
     the registry (pinned temp aspects_v1.json, never the live file).
  3. Residue — `## Review` fence harvested; notes persisted on this run's
     chain under unit='aspect_integration'; malformed == 0. For this unit
     the notes are the ONLY durable record of classification reasoning.

A settled brain has nothing unclassified, so the sim seeds a novel node type
and a novel edge relation FIRST; the decoder then finds them and builds real
example records (decoder-built input, production-faithful shape).

    ./dev python3 eval/sim_aspect_journal.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.isolated_brain import IsolatedBrain
from tests.interaction_override import override_interaction
from servers.trace_contract import (JOURNAL_REVIEW_INSTRUCTION,
                                    extract_review_block, parse_journal_notes)
from servers.scales.s2.aspect_decoder import AspectDecoder
from servers.scales.s2.aspect_encoder import AspectEncoder
from servers.scales.s2.aspect_contract import ASPECT
from servers.scales.s2 import base as base_mod


def make_v8(v: str) -> str:
    """→ v8: unpin the JSON-only line so it can't suppress the appended
    `## Review` fence. 'before it', not 'around it' — 'around' is
    bidirectional and still forbade the trailing review fence (the v7
    half-fix; code review a108cfc finding #3). IDEMPOTENT: applies only the
    edits whose anchors are present, so the probe keeps running against
    whatever version is active."""
    new_line = ("Use this exact shape (no markdown fences around the array, "
                "no prose before it):")
    out = v.replace(
        "Use this exact shape (no markdown fences, no prose):", new_line)
    out = out.replace(
        "Use this exact shape (no markdown fences around the array, "
        "no prose around it):", new_line)
    assert new_line in out, 'JSON-only line: no known form present'
    return out


def main():
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        print('isolated brain: %d non-archived nodes' % env.node_count())

        # 1. Derive the v8 candidate from the ACTIVE prompt (no-op when v8
        #    is already live) and activate it on the isolated brain.
        v6 = brain.get_interaction_prompt('s2_aspects') or ''
        assert v6, 'no s2_aspects prompt in isolated brain'
        v8 = make_v8(v6)
        params = brain.get_interaction_config('s2_aspects') or {}
        override_interaction(brain, 's2_aspects', template=v8,
                             parameters=params, set_by='eval:journal_v8')

        # 1b. Assembly check (deterministic) — exactly what _call_llm builds.
        enc0 = AspectEncoder(brain, config=ASPECT)
        asm = enc0.journal.decorate_system(v8, multi_round=False)
        checks = {
            'review block present':     JOURNAL_REVIEW_INSTRUCTION in asm,
            'no closure (single-shot)': '## Finishing' not in asm,
            'json-only line softened':  'no prose before it' in asm,
            'review block at tail':     asm.rstrip().endswith('Stay sharp.'),
        }
        print('\n=== assembly ===')
        for k, v in checks.items():
            print('  %s %s' % ('✓' if v else '✗', k))

        # 2. Seed novel strings (a settled brain has nothing unclassified),
        #    then decode — the decoder builds real example records for them.
        seed = brain.remember(
            type='sim_probe_type', title='journal probe: novel type instance',
            content='Probe node carrying a never-classified type so the '
                    'aspect decoder proposes it with a real example record.')
        anchor = brain.conn.execute(
            "SELECT id FROM nodes WHERE archived=0 AND id != ? LIMIT 1",
            (seed['id'],)).fetchone()[0]
        brain._graph.add_relation(
            seed['id'], anchor, relation='sim_probes_against',
            description='probe edge carrying a never-classified relation',
            encoding_source='eval:journal_v8')

        dec = AspectDecoder(brain, config=ASPECT).run()
        proposals = dec.get('proposals', [])
        print('\n=== decoder ===  proposals=%d  skipped=%s'
              % (len(proposals), dec.get('skipped')))
        for p in proposals:
            print('  %s "%s" ×%d' % (p['category'], p['value'], p['count']))
        if not proposals:
            print('decoder proposed nothing — seeding failed?')
            return

        # 3. REAL encode (one Sonnet call), raw response teed.
        raw_seen = []
        real_once = base_mod.run_llm_once

        def tee_run_llm_once(*a, **kw):
            raw, tel = real_once(*a, **kw)
            raw_seen.append(raw)
            return raw, tel

        base_mod.run_llm_once = tee_run_llm_once
        try:
            encoder = AspectEncoder(brain, config=ASPECT)
            print('\n--- encoder (REAL Sonnet run) ---')
            t0 = time.time()
            result = encoder.run(proposals) or {}
            dt = time.time() - t0
        finally:
            base_mod.run_llm_once = real_once
        print('classified=%d rejected=%d errors=%d  %.1fs'
              % (result.get('classified', 0), result.get('rejected', 0),
                 len(result.get('errors', [])), dt))
        for r in (result.get('rejected_details') or [])[:5]:
            print('  rejected: %s' % r.get('reason'))

        # 3b. residue — raw-response fence diagnosis + read-back on this chain.
        block = extract_review_block(raw_seen[0]) if raw_seen else None
        notes, mal = parse_journal_notes(block) if block is not None else ([], [])
        run_chain = encoder.chain_id()
        rows = [r for r in brain.journal_notes(
                    scale='s2', unit='aspect_integration', k=1)
                if r.get('chain_id') == run_chain]
        print('\n=== residue ===  fence=%s  parsed=%d  MALFORMED=%d  persisted(this run)=%d'
              % ('present' if block is not None else 'ABSENT',
                 len(notes), len(mal), len(rows)))
        for r in rows:
            print('  %s · %s · %s' % (
                (r.get('tag') or '—'), r.get('subject', ''), (r.get('note') or '')[:180]))

        # Verdict.
        print('\n=== verdict ===')
        assembly_ok = all(checks.values())
        payload_ok = result.get('classified', 0) > 0 and not result.get('errors')
        residue_ok = (block is not None and not mal and len(notes) == len(rows))
        print('  %s assembly correct (review in, closure/arc out, json line softened)'
              % ('✓' if assembly_ok else '✗'))
        print('  %s payload survives harvest (classifications parsed + merged, 0 errors)'
              % ('✓' if payload_ok else '✗'))
        print('  %s residue clean (fence present, 0 malformed, parsed==persisted)'
              % ('✓' if residue_ok else '⚠'))


if __name__ == '__main__':
    main()
