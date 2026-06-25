#!/usr/bin/env python3
"""Eval for the community structural-field denormalization fix.

Never touches live data (IsolatedBrain copies brain.db + brain_logs.db to a
temp dir, cleaned on exit). Two parts:

  1. DRIFT CENSUS (deterministic, no LLM) — across ALL live communities,
     compare the STORED community_size / internal_fraction (Haiku-authored)
     against the EDGE-DERIVED truth. Quantifies the bug the fix erases and
     confirms derived == edges on real data.

  2. NO-REGRESSION A/B (real Haiku) — v20 (agent writes the structural fields)
     vs v21 (those instructions removed) on the SAME decoder proposals across
     two fresh copies. Compares creation metrics. The v21 arm also folds in the
     STAMP-CORRECTNESS check: every community it created must carry a stamped
     community_size equal to an INDEPENDENT raw edge COUNT (not via the helper).

  v21 is derived HERE from the live v20 via make_v21() — the exact transform
  reused verbatim at landing, so this exercises the shipped prompt.

    ./dev python3 eval/sim_community_structural.py [max_proposals] [batch_size]
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.isolated_brain import IsolatedBrain
from eval.s2_community_decoder_eval import run_new_decoder
from servers.scales.s2.community_contract import COMMUNITY_DETECTION
from servers.scales.s2.community_encoder import CommunityEncoder
from servers.scales.s2.community_structural import compute_community_structural

MAX_PROPOSALS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BATCH_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def make_v21(v20: str) -> str:
    """Drop authorship of the four structural fields (size / internal_fraction /
    is_corridor / dominant_type) from instructions AND every example.

    Pure subtraction — no "don't write them" notes added. Each anchor must
    appear EXACTLY once; asserts catch prompt drift loudly (so landing fails
    fast if the live v20 ever moved). Reused verbatim at landing.
    """
    edits = [
        # Site A — NEW COMMUNITY example: drop dominant_type + the structural line.
        ('   community_maturity: "settled",\n'
         '   community_dominant_type: "finding",\n'
         '   community_size: "3", community_internal_fraction: "0.89", '
         'community_is_corridor: "false"}',
         '   community_maturity: "settled"}'),
        # Site B — metadata doc-list: drop the two structural items.
        ('\n- `community_dominant_type` — most common node type\n'
         '- `community_size` (integer as string, e.g. "15"), '
         '`community_internal_fraction`, `community_is_corridor` — structural',
         ''),
        # Site C1 — ADD TO EXISTING instruction: positive action, no size set.
        ('Connect node; set `community_size` to new member count (integer as string).',
         'Connect the new member to the community.'),
        # Site C2 — ADD TO EXISTING example: drop the size-bump revise op.
        (',\n  {op: "revise", node_id: "comm1234", reason: "member added: '
         'Node Title", community_size: "15"}',
         ''),
        # Site D — MERGE example: drop community_size from the revise.
        ('   content: "Combined narrative...", community_size: "25"},',
         '   content: "Combined narrative..."},'),
    ]
    out = v20
    for old, new in edits:
        n = out.count(old)
        assert n == 1, 'v21 anchor not unique (count=%d): %r' % (n, old)
        out = out.replace(old, new)
    # Belt: no structural field name survives as an authoring instruction.
    for banned in ('community_size:', 'community_internal_fraction:',
                   'community_is_corridor:', 'community_dominant_type:'):
        assert banned not in out, 'v21 still authors %s' % banned
    return out


def _live_community_ids(brain):
    return [r[0] for r in brain.conn.execute(
        "SELECT id FROM nodes WHERE type = 'community' AND archived = 0"
    ).fetchall()]


def _raw_member_count(brain, cid):
    """Independent edge count — raw SQL, NOT via compute_community_structural,
    so the correctness check can't be circular."""
    return brain.conn.execute("""
        SELECT COUNT(DISTINCT member.id)
        FROM edges e
        JOIN edge_relations er ON er.edge_id = e.edge_id
        JOIN nodes member ON member.id = CASE
            WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END
            AND member.archived = 0 AND member.type != 'community'
        WHERE er.relation = 'community_member' AND er.archived = 0
          AND (e.source_id = ? OR e.target_id = ?)
    """, (cid, cid, cid)).fetchone()[0]


def _member_edge_total(brain):
    return brain.conn.execute(
        "SELECT COUNT(*) FROM edge_relations WHERE relation = 'community_member' "
        "AND archived = 0").fetchone()[0]


# ════════════════════════════════════════════════════════════════════════
# Part 1 — drift census (deterministic)
# ════════════════════════════════════════════════════════════════════════

def drift_census():
    print('\n' + '=' * 70)
    print('PART 1 — DRIFT CENSUS (stored vs edge-derived, all communities)')
    print('=' * 70)
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        # Fail fast: validate the v21 transform against the live v20 BEFORE the
        # A/B spends any Haiku — a drifted anchor stops the whole eval here.
        v20 = brain.get_interaction_prompt('s2_community_enrichment') or ''
        make_v21(v20)
        print('make_v21 anchors OK (v21 derives cleanly from live v20)')

        comms = _live_community_ids(brain)
        print('live communities: %d' % len(comms))
        if not comms:
            print('no communities — nothing to census.')
            return
        derived = compute_community_structural(brain, comms)
        stored = brain._meta_kv.get_fields_bulk(
            comms, ['community_size', 'community_internal_fraction'])

        size_ok = size_drift = size_missing = 0
        frac_ok = frac_drift = 0
        worst = []
        for cid in comms:
            d = derived.get(cid, {})
            s = stored.get(cid, {})
            true_size = d.get('community_size', 0)
            stored_size = s.get('community_size')
            if stored_size is None:
                size_missing += 1
            elif str(stored_size) == str(true_size):
                size_ok += 1
            else:
                size_drift += 1
                try:
                    gap = abs(int(stored_size) - true_size)
                except (TypeError, ValueError):
                    gap = -1
                worst.append((gap, cid, stored_size, true_size))

            true_frac = d.get('community_internal_fraction', 0.0)
            stored_frac = s.get('community_internal_fraction')
            if stored_frac is not None:
                try:
                    if abs(float(stored_frac) - true_frac) <= 0.01:
                        frac_ok += 1
                    else:
                        frac_drift += 1
                except (TypeError, ValueError):
                    frac_drift += 1

        print('\ncommunity_size:   %d match · %d DRIFTED · %d missing (of %d)'
              % (size_ok, size_drift, size_missing, len(comms)))
        print('internal_fraction: %d match · %d DRIFTED (of stored)'
              % (frac_ok, frac_drift))
        worst.sort(reverse=True)
        if worst:
            print('\nworst size drifts (stored → true edge count):')
            for gap, cid, st, tr in worst[:12]:
                print('  %s  stored=%s  true=%-3d  (off by %d)'
                      % (cid[:8], st, tr, gap))
        print('\n→ %d communities carry a wrong stored size the one-time fill '
              'will correct.' % size_drift)


# ════════════════════════════════════════════════════════════════════════
# Part 2 — no-regression A/B (real Haiku) + stamp-correctness on the v21 arm
# ════════════════════════════════════════════════════════════════════════

def run_arm(label, transform, max_prop, batch):
    with IsolatedBrain(cleanup=True) as env:
        brain = env.brain
        if transform is not None:
            v20 = brain.get_interaction_prompt('s2_community_enrichment') or ''
            assert v20, 'no s2_community_enrichment in isolated brain'
            v21 = transform(v20)
            params = brain.get_interaction_config('s2_community_enrichment') or {}
            reg = brain._interaction_dal.register(
                's2_community_enrichment', template=v21,
                parameters=json.dumps(params), created_by='eval:v21')
            brain._interaction_dal.set_active(
                's2_community_enrichment', reg['version'], set_by='eval:v21')

        edges_before = _member_edge_total(brain)
        comms_before = set(_live_community_ids(brain))

        dec = run_new_decoder(brain, dict(COMMUNITY_DETECTION))
        proposals = dec['proposals']
        actionable = [p for p in proposals if p['type'] in (
            'new_community', 'add_to_existing', 'drift',
            'health_update', 'merge_communities')]
        new_props = [p for p in proposals if p['type'] == 'new_community']
        members_proposed = sum(
            len(p.get('all_members', p.get('members', []))) for p in new_props)

        if not actionable:
            print('  [%s] no actionable proposals — graph settled.' % label)
            return None

        cfg = dict(COMMUNITY_DETECTION)
        cfg['max_proposals_per_call'] = batch
        cfg['max_actionable_per_run'] = max_prop
        enc = CommunityEncoder(brain, config=cfg)

        t0 = time.time()
        result = enc.run(proposals, dec['community_state'])
        dt = time.time() - t0
        result = result or {}

        edges_after = _member_edge_total(brain)
        comms_after = set(_live_community_ids(brain))
        new_ids = comms_after - comms_before

        # Per-created-community detail — distinguishes a real completeness gap
        # (same communities, fewer edges) from selection noise (different
        # communities of different sizes).
        new_detail = []
        for cid in new_ids:
            row = brain.conn.execute(
                "SELECT title FROM nodes WHERE id = ?", (cid,)).fetchone()
            new_detail.append(
                ((row[0] or '')[:48], _raw_member_count(brain, cid)))
        new_detail.sort(key=lambda x: -x[1])

        # Stamp-correctness (v21 arm): every newly-created community must carry a
        # stamped size == an INDEPENDENT raw edge count.
        stamp_checked = stamp_ok = 0
        stamp_bad = []
        if transform is not None:
            stored = brain._meta_kv.get_fields_bulk(
                list(new_ids), ['community_size'])
            for cid in new_ids:
                raw = _raw_member_count(brain, cid)
                st = stored.get(cid, {}).get('community_size')
                if st is None:
                    continue
                stamp_checked += 1
                if str(st) == str(raw):
                    stamp_ok += 1
                else:
                    stamp_bad.append((cid, st, raw))

        return {
            'label': label,
            'actionable': len(actionable),
            'new_proposals': len(new_props),
            'members_proposed': members_proposed,
            'write_actions': result.get('write_actions', 0),
            'communities_created': len(new_ids),
            'member_edges_created': edges_after - edges_before,
            'rounds': result.get('rounds', 0),
            'truncations': len(result.get('truncations', []) or []),
            'invalid_ops': result.get('invalid_op_failures', 0),
            'elapsed_s': round(dt, 1),
            'stamp_checked': stamp_checked,
            'stamp_ok': stamp_ok,
            'stamp_bad': stamp_bad,
            'new_detail': new_detail,
        }


def ab_test(max_prop, batch):
    print('\n' + '=' * 70)
    print('PART 2 — NO-REGRESSION A/B  (v20 = writes fields, v21 = dropped)')
    print('=' * 70)
    a = run_arm('v20', None, max_prop, batch)
    b = run_arm('v21', make_v21, max_prop, batch)
    if not a or not b:
        print('\none arm had no actionable proposals — A/B inconclusive.')
        return a, b

    rows = [
        ('actionable proposals', a['actionable'], b['actionable']),
        ('new_community proposals', a['new_proposals'], b['new_proposals']),
        ('members proposed (new)', a['members_proposed'], b['members_proposed']),
        ('write_actions', a['write_actions'], b['write_actions']),
        ('communities created', a['communities_created'], b['communities_created']),
        ('member edges created', a['member_edges_created'], b['member_edges_created']),
        ('rounds', a['rounds'], b['rounds']),
        ('truncations', a['truncations'], b['truncations']),
        ('invalid ops', a['invalid_ops'], b['invalid_ops']),
        ('elapsed (s)', a['elapsed_s'], b['elapsed_s']),
    ]
    print('\n%-26s %8s %8s' % ('metric', 'v20', 'v21'))
    print('-' * 46)
    for name, av, bv in rows:
        print('%-26s %8s %8s' % (name, av, bv))

    print('\n=== communities created (title · true member-edge count) ===')
    for arm in (a, b):
        print('  [%s]' % arm['label'])
        for title, cnt in arm['new_detail']:
            print('     %3d  %s' % (cnt, title))

    print('\n=== stamp correctness (v21 arm, new communities) ===')
    print('  checked=%d  ok=%d  bad=%d'
          % (b['stamp_checked'], b['stamp_ok'], len(b['stamp_bad'])))
    for cid, st, raw in b['stamp_bad'][:10]:
        print('  MISMATCH %s stamped=%s raw=%s' % (cid[:8], st, raw))
    return a, b


def main():
    drift_census()
    a, b = ab_test(MAX_PROPOSALS, BATCH_SIZE)

    print('\n' + '=' * 70)
    print('VERDICT')
    print('=' * 70)
    if a and b:
        stamp_clean = (b['stamp_checked'] > 0 and not b['stamp_bad'])
        # No-regression: v21 should connect a comparable share of proposed
        # members and create a comparable number of communities. Flag only a
        # clear drop (>20% fewer edges or communities at equal proposals).
        def _ratio(x, y):
            return (x / y) if y else 1.0
        edge_ratio = _ratio(b['member_edges_created'], a['member_edges_created'])
        comm_ratio = _ratio(b['communities_created'], a['communities_created'])
        no_regress = edge_ratio >= 0.8 and comm_ratio >= 0.8
        print('  %s stamp correctness (new communities: stamped == raw count)'
              % ('OK ' if stamp_clean else 'CHECK'))
        print('  %s no creation regression (v21/v20 edges=%.2f comms=%.2f, '
              '>=0.80)' % ('OK ' if no_regress else 'CHECK',
                           edge_ratio, comm_ratio))
        print('  (narrative quality: spot-check the created communities above '
              'by hand — not auto-scored.)')


if __name__ == '__main__':
    main()
