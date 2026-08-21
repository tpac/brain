#!/usr/bin/env python3
"""Run a REAL healer cycle on an IsolatedBrain copy with the v5 prompt + the
journal binding on _call_llm (Phase 3: single-shot units join the note
contract), and observe:

  1. Assembly — the decorated system prompt carries the review block at the
     tail, NO closure (single-shot has no terminal-turn ambiguity), NO arc,
     and NO legacy `## When Done` HEALED/PATTERNS journal (the dead-promise
     text v5 removes).
  2. Payload survival — Haiku's JSON healings parse after the harvest strip
     (the rfind hazard), fields actually written.
  3. Residue — `## Review` fence harvested; notes persisted on this run's
     chain under unit='healer'; malformed == 0.

Never touches live data (IsolatedBrain copies brain.db + brain_logs.db and
pins ASPECTS_JSON_PATH into a temp dir). v5 is derived HERE from the live v4
via make_v5() — the exact transform reused at landing — and activated on the
isolated brain only.

    ./dev python3 eval/sim_healer_journal.py [n_nodes]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.isolated_brain import IsolatedBrain
from servers.trace_contract import (JOURNAL_REVIEW_INSTRUCTION,
                                    extract_review_block, parse_journal_notes)
from servers.scales.s2.healer_decoder import HealerDecoder
from servers.scales.s2.healer_encoder import HealerEncoder
from servers.scales.s2.healer_contract import HEALER
from servers.scales.s2 import base as base_mod

N_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def make_v5(v4: str) -> str:
    """v4 → v5: delete the legacy `## When Done` journal section (the
    HEALED/SKIPPED/PATTERNS/WATCHING block whose output the code discards —
    the dead promise; residue now rides the runtime review block), and
    unpin the JSON-only line so it can't fight the appended `## Review`
    fence. IDEMPOTENT: each edit applies only when its anchor is present,
    so the probe keeps running after the transform lands (v5 active) —
    post-landing it's a no-op and the sim probes the ACTIVE prompt as-is."""
    old_line = "No markdown fences, no explanation, just the JSON array."
    new_line = ("No markdown fences around the array, no explanation "
                "before it — just the JSON array.")
    out = v4.replace(old_line, new_line)
    assert new_line in out, 'JSON-only line: neither v4 nor v5 form present'

    j = out.find("## When Done")
    return out[:j].rstrip() if j != -1 else out


def main():
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        print('isolated brain: %d non-archived nodes' % env.node_count())

        # 1. Derive v5 from the live v4 and activate it on the isolated brain.
        v4 = brain.get_interaction_prompt('s2_healer') or ''
        assert v4, 'no s2_healer prompt in isolated brain'
        v5 = make_v5(v4)
        params = brain.get_interaction_config('s2_healer') or {}
        reg = brain._interaction_dal.register(
            's2_healer', template=v5,
            parameters=json.dumps(params), created_by='eval:journal_v5')
        brain._interaction_dal.set_active(
            's2_healer', reg['version'], set_by='eval:journal_v5')

        # 1b. Assembly check (deterministic) — exactly what _call_llm builds.
        cfg = dict(HEALER)
        cfg['max_nodes_per_run'] = N_NODES
        cfg['max_nodes_per_call'] = N_NODES
        enc0 = HealerEncoder(brain, config=cfg)
        asm = enc0.journal.decorate_system(v5, multi_round=False)
        checks = {
            'review block present':     JOURNAL_REVIEW_INSTRUCTION in asm,
            'no closure (single-shot)': '## Finishing' not in asm,
            'no arc':                   '## Arc' not in asm,
            'legacy journal gone':      'PATTERNS:' not in asm and '## When Done' not in asm,
            'review block at tail':     asm.rstrip().endswith('Stay sharp.'),
        }
        print('\n=== assembly ===')
        for k, v in checks.items():
            print('  %s %s' % ('✓' if v else '✗', k))

        # 2. Decode — real gap scan on the production copy.
        dec = HealerDecoder(brain, config=cfg).run()
        proposals = dec.get('proposals', [])
        print('\n=== decoder ===  proposals=%d  stats=%s  skipped=%s'
              % (len(proposals), dec.get('stats'), dec.get('skipped')))
        if not proposals:
            print('no healing gaps on the copy — nothing to probe this run.')
            return

        # 3. REAL encode (one batch → one Haiku call), raw response teed so
        #    fence presence is diagnosable even when notes come out 0.
        raw_seen = []
        real_once = base_mod.run_llm_once

        def tee_run_llm_once(*a, **kw):
            raw, tel = real_once(*a, **kw)
            raw_seen.append(raw)
            return raw, tel

        base_mod.run_llm_once = tee_run_llm_once
        try:
            encoder = HealerEncoder(brain, config=cfg)
            print('\n--- encoder (REAL Haiku run) ---')
            t0 = time.time()
            result = encoder.run(proposals[:N_NODES]) or {}
            dt = time.time() - t0
        finally:
            base_mod.run_llm_once = real_once
        print('healed=%d fields=%d skipped=%d errors=%d  %.1fs'
              % (result.get('nodes_healed', 0), result.get('fields_written', 0),
                 result.get('skipped', 0), len(result.get('errors', [])), dt))

        # 3b. residue — raw-response fence diagnosis + authoritative read-back
        #     for THIS run's chain.
        wf = mal = sections = 0
        for raw in raw_seen:
            block = extract_review_block(raw)
            if block is None:
                continue
            sections += 1
            notes, m = parse_journal_notes(block)
            wf += len(notes)
            mal += len(m)
        run_chain = encoder.chain_id()
        rows = [r for r in brain.journal_notes(scale='s2', unit='healer', k=1)
                if r.get('chain_id') == run_chain]
        print('\n=== residue ===  llm_calls=%d  review-sections=%d  parsed=%d  '
              'MALFORMED=%d  persisted(this run)=%d'
              % (len(raw_seen), sections, wf, mal, len(rows)))
        for r in rows:
            print('  %s · %s · %s' % (
                (r.get('tag') or '—'), r.get('subject', ''), (r.get('note') or '')[:180]))

        # Verdict.
        print('\n=== verdict ===')
        assembly_ok = all(checks.values())
        payload_ok = result.get('nodes_healed', 0) > 0 and not result.get('errors')
        residue_ok = (sections == len(raw_seen) and mal == 0 and wf == len(rows))
        print('  %s assembly correct (review in, closure/arc/legacy out)'
              % ('✓' if assembly_ok else '✗'))
        print('  %s payload survives harvest (healings parsed + written, 0 errors)'
              % ('✓' if payload_ok else '✗'))
        print('  %s residue clean (fence on every call, 0 malformed, parsed==persisted)'
              % ('✓' if residue_ok else '⚠'))


if __name__ == '__main__':
    main()
