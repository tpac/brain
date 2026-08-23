#!/usr/bin/env python3
"""One-shot: run a REAL community cycle (decoder → encoder → rejection stamping)
against an IsolatedBrain copy with the Phase-5 v20 prompt, and observe BOTH the
residue review notes the encoder writes AND that rejection fingerprinting still
works after the journal-line reject channel is gone.

Never touches live data (IsolatedBrain copies brain.db + brain_logs.db to a temp
dir, cleaned on exit). The v20 prompt (journal section removed, reject reframed
to "no action") is derived HERE from the live v19 via make_v20() — the exact
transform reused at landing — and activated on the isolated brain, so this
exercises the full landing path end-to-end with a real Haiku encode.

Three things validated:
  1. Residue — encoder emits `## Review`; notes parse clean; persisted == parsed;
     malformed == 0 (the `·`-delimiter fidelity check, now on community).
  2. Suppression survives — s2_rejections still grows for genuinely-skipped
     proposals (the load-bearing proof that dropping the reject journal-line did
     NOT break community's structural rejection mechanism).
  3. Sane split — the encoder neither acts on everything nor passes on
     everything without the old forced "REJECTED:" articulation.

    ./dev python3 eval/sim_community_journal.py [max_proposals] [batch_size]
"""
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.isolated_brain import IsolatedBrain
from tests.interaction_override import override_interaction
from servers.trace_contract import extract_review_block, parse_journal_notes
from eval.s2_community_decoder_eval import run_decoder
from servers.scales.s2.community_contract import COMMUNITY_DETECTION
from servers.scales.s2.community_encoder import CommunityEncoder

MAX_PROPOSALS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 6  # small → force multi-batch


def make_v20(v19: str) -> str:
    """Apply the 5 approved Phase-5 edits + delete the `## Journal` section.

    Each text anchor must appear EXACTLY once — asserts catch prompt drift loudly
    (so this fails fast at landing if the live v19 ever moved). Reused verbatim
    against the live v19 when landing the prompt flip.
    """
    edits = [
        ("Every proposal gets a decision: accept (tool call) or reject (journal line).",
         "Every proposal gets a decision: accept (tool call) or reject (just don't act — it's recorded for you)."),
        ("then brain_batch, then journal. DONE.",
         "then brain_batch."),
        ("→ KEEP it: reject in the journal with a one-line reason.",
         "→ KEEP it: reject (no action)."),
        ("If you're uncertain about a proposal, reject it in the journal with a one-line reason.",
         "If you're uncertain about a proposal, reject it (no action)."),
        ("Accept → tool call. Reject → journal line.",
         "Accept → tool call. Reject → no action."),
    ]
    out = v19
    for old, new in edits:
        n = out.count(old)
        assert n == 1, 'v20 anchor not unique (count=%d): %r' % (n, old)
        out = out.replace(old, new)
    # Delete the `## Journal (after ALL tool calls)` section: from its heading up
    # to the next heading (`## YOUR ROLE`). It sits mid-prompt with YOUR ROLE
    # after it, which is why it can't be stripped-to-end like consolidation's.
    j = out.find("## Journal (after ALL tool calls)")
    y = out.find("## YOUR ROLE")
    assert j != -1 and y != -1 and j < y, 'journal/your-role anchors bad: j=%d y=%d' % (j, y)
    return out[:j].rstrip() + "\n\n" + out[y:]


def _parse_per_batch(final_text):
    """Re-parse the encoder's concatenated multi-batch final_text the way the
    per-batch write_journal_notes did — split on the `--- batch N ---` marker,
    extract each chunk's `## Review` fence, parse. Returns
    (well_formed, malformed, sections) where `sections` counts batches that
    emitted a `## Review` section at all (empty fence included) — so the verdict
    can tell a genuinely-clean run from drift where no review was ever emitted.
    """
    chunks = re.split(r'\n--- batch \d+ ---\n', final_text)
    well_formed, malformed, sections = 0, 0, 0
    for chunk in chunks:
        if not chunk.strip():
            continue
        block = extract_review_block(chunk)
        if block is None:      # no `## Review` / broken fence — not a section
            continue
        sections += 1          # block == '' (clean empty) or fence content
        notes, mal = parse_journal_notes(block)
        well_formed += len(notes)
        malformed += len(mal)
    return well_formed, malformed, sections


def main():
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        print('isolated brain: %d non-archived nodes' % env.node_count())

        # 1. Derive v20 from the live v19 and activate it on the isolated brain.
        v19 = brain.get_interaction_prompt('s2_community_enrichment') or ''
        assert v19, 'no s2_community_enrichment prompt in isolated brain'
        v20 = make_v20(v19)
        params = brain.get_interaction_config('s2_community_enrichment') or {}
        override_interaction(brain, 's2_community_enrichment', template=v20,
                             parameters=params, set_by='eval:journal_port_v20')
        active = brain.get_interaction_prompt('s2_community_enrichment') or ''
        print('\n=== v20 prompt checks ===')
        print('  journal section removed : %s' % ('## Journal (after ALL tool calls)' not in active))
        print('  reject→no-action        : %s' % ('Reject → no action' in active))
        print('  reject (just dont act)  : %s' % ("reject (just don't act" in active))
        print('  review NOT baked (runtime-injected): %s' % ('## Review' not in active))
        print('  ends at YOUR ROLE       : %s' % active.rstrip().endswith('permanently lost.'))

        rej_before = brain.conn.execute("SELECT COUNT(*) FROM s2_rejections").fetchone()[0]

        # 2. Real decode → encode cycle (encoder.run does per-batch
        #    write_journal_notes + record_rejections internally).
        print('\n--- decoder (cold scan) ---')
        dec = run_decoder(brain, dict(COMMUNITY_DETECTION))
        proposals = dec['proposals']
        actionable = [p for p in proposals if p['type'] in (
            'new_community', 'add_to_existing', 'drift', 'health_update', 'merge_communities')]
        print('surviving proposals: %d (actionable: %d)' % (len(proposals), len(actionable)))
        if not actionable:
            print('\nno actionable proposals — community graph settled; '
                  'nothing to encode/observe this run.')
            return

        cfg = dict(COMMUNITY_DETECTION)
        cfg['max_proposals_per_call'] = BATCH_SIZE        # small → multi-batch
        cfg['max_actionable_per_run'] = MAX_PROPOSALS
        encoder = CommunityEncoder(brain, config=cfg)

        print('\n--- encoder (REAL run: inject review block + encode + '
              'per-batch write_journal_notes + record_rejections) ---')
        t0 = time.time()
        result = encoder.run(proposals, dec['community_state'])
        dt = time.time() - t0
        if not result:
            print('encoder returned None.')
            return
        final_text = result.get('final_text', '') or ''
        n_batches = final_text.count('--- batch ')
        print('actions=%d writes=%d rounds=%d batches=%d  %.1fs' % (
            result.get('actions', 0), result.get('write_actions', 0),
            result.get('rounds', 0), n_batches, dt))

        # 3a. Residue: re-parse per batch (mirrors the per-batch write), then the
        #     authoritative read-back of what actually persisted FOR THIS RUN.
        #     journal_notes returns the last K runs (K=3 for community) and
        #     IsolatedBrain copies live brain_logs.db — so filter to this run's
        #     chain_id, else prior community runs' notes would inflate the count
        #     and break the parsed==persisted check.
        wf, mal, sections = _parse_per_batch(final_text)
        run_chain = encoder.chain_id()
        rows = [r for r in brain.journal_notes(scale='s2', unit='community_detection', k=1)
                if r.get('chain_id') == run_chain]
        print('\n=== residue ===  review-sections=%d  parsed(well-formed)=%d  MALFORMED=%d  persisted(this run)=%d'
              % (sections, wf, mal, len(rows)))
        for r in rows:
            print('  %s · %s · %s' % (
                (r.get('tag') or '—'), r.get('subject', ''), (r.get('note') or '')[:160]))

        # 3b. Suppression: did record_rejections still stamp the skipped proposals?
        rej_after = brain.conn.execute("SELECT COUNT(*) FROM s2_rejections").fetchone()[0]
        skipped = result.get('rejection_skipped_count', 0)
        print('\n=== suppression ===  s2_rejections: %d → %d (Δ%+d)  | encoder-reported skipped=%d'
              % (rej_before, rej_after, rej_after - rej_before, skipped))

        # 3c. Sane split.
        sent = result.get('proposals_sent') or len(actionable)
        print('\n=== accept/reject split ===  actionable-sent≈%d  write_actions=%d  skipped=%d'
              % (sent, result.get('write_actions', 0), skipped))

        # Verdict.
        print('\n=== verdict ===')
        # sections>0 guards a vacuous 0==0 pass (encoder emitted no ## Review).
        residue_ok = (sections > 0 and mal == 0 and wf == len(rows))
        supp_ok = (skipped == 0) or (rej_after > rej_before)
        print('  %s residue clean (parsed==persisted, 0 malformed)' % ('✓' if residue_ok else '✗'))
        print('  %s suppression intact (skipped → s2_rejections grew, or nothing skipped)'
              % ('✓' if supp_ok else '✗'))
        print('  %s ran (≥1 write OR a deliberate all-skip)'
              % ('✓' if result.get('rounds', 0) > 0 else '✗'))


if __name__ == '__main__':
    main()
