#!/usr/bin/env python3
"""A/B the community encoder MODEL on isolated production copies.

One invocation = one arm-run: copy the (frozen) source brain, decode with
the production decoder (eval seam run_decoder), run the REAL
CommunityEncoder with only `model` overridden, then report deltas the
model choice owns:

  completion   — write_actions / proposals sent, rounds, communities created
  edge_omission— membership-backfill firings during the run (the restorer
                 healing communities the encoder declared but left edgeless)
  journal      — persisted journal notes for this run's chain
  discipline   — rejections stamped, brain_batch invalid ops
  quality      — per-created-community field presence (question, situation,
                 latest_development) + sizes, for the judge pass
  cost         — tokens in/out, wall-clock

Run arms in parallel with PYTHONHASHSEED=0 and a shared frozen --source-dir
so every arm decodes identical proposals:

    cp brain.db brain_logs.db <master>/
    PYTHONHASHSEED=0 ./dev python3 eval/ab_community_model.py \
        --model claude-haiku-4-5-20251001 --label haiku_1 \
        --source-dir <master> --out <reports>/haiku_1.json &
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eval.s2_community_decoder_eval import run_decoder  # noqa: E402
from servers.scales.s2.community_contract import COMMUNITY_DETECTION  # noqa: E402

ACTIONABLE = ('new_community', 'add_to_existing', 'drift',
              'health_update', 'merge_communities')


def _live_communities(brain):
    return {r[0] for r in brain.conn.execute(
        "SELECT id FROM nodes WHERE type='community' AND archived=0 "
        "AND encoding_source='s2:community_detection'").fetchall()}


def _member_edges(brain):
    return brain.conn.execute(
        "SELECT COUNT(*) FROM edge_relations "
        "WHERE relation='community_member' AND archived=0").fetchone()[0]


def _rejections(brain):
    return brain.conn.execute(
        "SELECT COUNT(*) FROM s2_rejections").fetchone()[0]


def _log_max_id(brain):
    return brain.logs_conn.execute(
        "SELECT COALESCE(MAX(id),0) FROM debug_log").fetchone()[0]


def _log_deltas(brain, since_id):
    rows = brain.logs_conn.execute(
        "SELECT source, event_type, metadata FROM debug_log "
        "WHERE id > ? AND source IN "
        "('community_membership_backfilled','brain_batch_invalid_op')",
        (since_id,)).fetchall()
    backfills, invalid_ops = [], 0
    for source, _etype, metadata in rows:
        if source == 'community_membership_backfilled':
            try:
                backfills.append(json.loads(metadata).get('context', ''))
            except (json.JSONDecodeError, TypeError):
                backfills.append(str(metadata)[:200])
        else:
            invalid_ops += 1
    return backfills, invalid_ops


def _community_quality(brain, comm_ids):
    out = []
    for cid in sorted(comm_ids):
        row = brain.conn.execute(
            "SELECT title, LENGTH(content) FROM nodes WHERE id=?",
            (cid,)).fetchone()
        meta = dict(brain.conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id=?",
            (cid,)).fetchall())
        edges = brain.conn.execute(
            "SELECT COUNT(*) FROM edges e JOIN edge_relations er "
            "ON er.edge_id=e.edge_id WHERE er.relation='community_member' "
            "AND er.archived=0 AND (e.source_id=? OR e.target_id=?)",
            (cid, cid)).fetchone()[0]
        declared = meta.get('community_members', '')
        out.append({
            'id': cid,
            'title': (row[0] if row else '?')[:80],
            'content_len': row[1] if row else 0,
            'has_question': bool(meta.get('question')),
            'has_situation': bool(meta.get('situation')),
            'has_latest_development': bool(
                meta.get('community_latest_development')),
            'declared_members': declared.count(',') + 1 if declared else 0,
            'member_edges': edges,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--source-dir', required=True,
                    help='frozen dir holding brain.db + brain_logs.db')
    ap.add_argument('--out', required=True, help='JSON report path')
    args = ap.parse_args()

    from tests.isolated_brain import IsolatedBrain

    report = {'label': args.label, 'model': args.model}
    with IsolatedBrain(production_dir=args.source_dir, cleanup=True) as env:
        brain = env.brain

        # The encoder resolves model DB-FIRST: interaction parameters beat
        # the config= dict (community_encoder.py `config.get('model',
        # self.config.get(...))`). A config-only override silently runs the
        # DB model — stamp the arm's model into THIS copy's interaction
        # parameters and verify, or the A/B measures nothing.
        params = dict(brain.get_interaction_config(
            's2_community_enrichment') or {})
        params['model'] = args.model
        tmpl = brain.get_interaction_prompt('s2_community_enrichment') or ''
        reg = brain._interaction_dal.register(
            's2_community_enrichment', template=tmpl,
            parameters=json.dumps(params), created_by='eval:ab_model')
        brain._interaction_dal.set_active(
            's2_community_enrichment', reg['version'], set_by='eval:ab_model')
        effective = (brain.get_interaction_config(
            's2_community_enrichment') or {}).get('model')
        report['model_effective'] = effective
        if effective != args.model:
            report['error'] = 'model stamp failed: effective=%s' % effective
            _write(args.out, report)
            sys.exit(2)

        dec = run_decoder(brain, dict(COMMUNITY_DETECTION))
        proposals = dec['proposals']
        actionable = [p for p in proposals if p.get('type') in ACTIONABLE]
        report['decode'] = {
            'proposals': len(proposals),
            'actionable': len(actionable),
            'by_type': dec['stats'].get('by_type', {}),
            'skipped': dec.get('skipped'),
        }
        if not actionable:
            report['error'] = 'no actionable proposals — arm is a no-op'
            _write(args.out, report)
            return

        comms_before = _live_communities(brain)
        edges_before = _member_edges(brain)
        rej_before = _rejections(brain)
        log_mark = _log_max_id(brain)

        from servers.scales.s2.community_encoder import CommunityEncoder
        cfg = dict(COMMUNITY_DETECTION)
        cfg['model'] = args.model
        encoder = CommunityEncoder(brain, config=cfg)

        t0 = time.time()
        result = encoder.run(proposals, dec['community_state']) or {}
        wall_s = round(time.time() - t0, 1)

        comms_after = _live_communities(brain)
        new_comms = comms_after - comms_before
        backfills, invalid_ops = _log_deltas(brain, log_mark)

        run_chain = encoder.chain_id()
        journal_rows = [r for r in brain.journal_notes(
            scale='s2', unit='community_detection', k=1)
            if r.get('chain_id') == run_chain]

        report.update({
            'completion': {
                'rounds': result.get('rounds', 0),
                'actions': result.get('actions', 0),
                'write_actions': result.get('write_actions', 0),
                'proposals_sent': result.get('proposals_sent'),
                'rejection_skipped': result.get('rejection_skipped_count', 0),
                'communities_created': len(new_comms),
                'member_edges_delta': _member_edges(brain) - edges_before,
            },
            'edge_omission': {
                'backfill_events': len(backfills),
                'backfill_details': backfills,
            },
            'discipline': {
                'rejections_stamped': _rejections(brain) - rej_before,
                'invalid_ops': invalid_ops,
            },
            'journal': {
                'persisted_notes': len(journal_rows),
                'notes': [{'tag': r.get('tag'),
                           'note': (r.get('note') or '')[:160]}
                          for r in journal_rows],
            },
            'quality': _community_quality(brain, new_comms),
            'cost': {
                'input_tokens': result.get('input_tokens', 0),
                'output_tokens': result.get('output_tokens', 0),
                'wall_s': wall_s,
            },
        })

    _write(args.out, report)


def _write(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    c = report.get('completion', {})
    print('[%s] %s: writes=%s created=%s backfills=%s notes=%s (%ss)' % (
        report['label'], report['model'],
        c.get('write_actions'), c.get('communities_created'),
        report.get('edge_omission', {}).get('backfill_events'),
        report.get('journal', {}).get('persisted_notes'),
        report.get('cost', {}).get('wall_s')), flush=True)


if __name__ == '__main__':
    main()
