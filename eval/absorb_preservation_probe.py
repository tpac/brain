#!/usr/bin/env python3
"""Absorb preservation probe — audit a merge for information loss across ALL
transfer dimensions at once.

`tests/test_absorb.py` unit-tests each dimension in isolation. This probe is the
*integration* guarantee + a reusable auditor: snapshot a node's full state, run
ONE absorb on a richly-populated fixture (source_refs + voice quotes + an
emergent KV field + real access_count + multiple external edges), then audit
that every dimension survived together. The same `snapshot_pre` / `audit` pair
points at a LIVE merge (S2-ABSORB-OP-DESIGN.md roadmap step 3) — snapshot the
real pair, absorb, audit, before trusting the prompt in production.

Dimensions audited:
  source_refs · access_count · kv_fill · kv_survivor_wins · content ·
  edges_migrated · archived_provenance · edge_direction_fidelity (diagnostic)

`edge_direction_fidelity` checks that absorb MIGRATES each edge's stored
direction faithfully (a regression guard on the migration loop). It is a
DIAGNOSTIC, never flips `lossless`. NOTE: the *stored* direction itself is
frequently set by accident upstream — `remember()`'s auto_connect creates
`co_accessed` edges (source = the newer node) between temporally-adjacent
nodes, and any later semantic `add_relation` on that pair inherits that
direction via `get_edge_id` (one physical direction per pair, v22). Fixing
THAT (creation-time direction) is the deferred edge-direction model work; it is
not absorb's job, and this probe does not gate on it.

Usage:
    ./dev python3 eval/absorb_preservation_probe.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ──────────────────────────────────────────────────────────────────
# Reusable auditor — dispatch-agnostic, inspects before/after state.
# ──────────────────────────────────────────────────────────────────

def _access_count(brain, node_id):
    row = brain.conn.execute(
        "SELECT access_count FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row[0] if row else 0


def _content(brain, node_id):
    row = brain.conn.execute(
        "SELECT content FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row[0] if row else None


def _external_edges(brain, node_id, exclude_id):
    """[(neighbor, relation, direction)] for node_id, excluding the intra-pair
    edge to exclude_id. Noise relations excluded by get_connections_bulk default."""
    conns = brain._graph.get_connections_bulk([node_id]).get(node_id, [])
    out = []
    for c in conns:
        if c['id'] == exclude_id:
            continue
        for rel in c.get('relations', []):
            out.append((c['id'], rel['relation'], c['direction']))
    return out


def _non_sys_kv(brain, node_id):
    kv = brain._meta_kv.get_all_bulk([node_id]).get(node_id, {})
    return {k: v for k, v in kv.items() if not k.startswith('_sys_')}


def snapshot_pre(brain, survivor_id, absorbed_id):
    """Capture full pre-merge state of both nodes. Call BEFORE absorb()."""
    return {
        'survivor_refs': set(brain._source_refs.get_source_refs(survivor_id)),
        'absorbed_refs': set(brain._source_refs.get_source_refs(absorbed_id)),
        'survivor_access': _access_count(brain, survivor_id),
        'absorbed_access': _access_count(brain, absorbed_id),
        'survivor_kv': _non_sys_kv(brain, survivor_id),
        'absorbed_kv': _non_sys_kv(brain, absorbed_id),
        'survivor_content': _content(brain, survivor_id),
        'absorbed_edges': _external_edges(brain, absorbed_id, survivor_id),
        'survivor_edges': set(_external_edges(brain, survivor_id, absorbed_id)),
    }


def audit(brain, survivor_id, absorbed_id, pre,
          prune_edges=(), drop_fields=(), overrides=()):
    """Audit a completed absorb. Returns {dimensions, lossless, warnings}.

    `lossless` is True when every HARD dimension passed; `edge_direction_fidelity`
    is diagnostic and never flips `lossless` (deferred model gap).
    `overrides` = field names the caller deliberately changed (excluded from the
    survivor-wins check, since an override legitimately replaces the value).
    """
    prune = set(prune_edges)
    drop = set(drop_fields)
    over = set(overrides)
    dims = {}

    # 1. source_refs — union, nothing lost.
    expected = pre['survivor_refs'] | pre['absorbed_refs']
    after_refs = set(brain._source_refs.get_source_refs(survivor_id))
    lost = expected - after_refs
    dims['source_refs'] = {
        'ok': not lost,
        'detail': 'expected %d refs, lost %s' % (len(expected), sorted(lost) or 'none'),
    }

    # 2. access_count — additive sum.
    after_access = _access_count(brain, survivor_id)
    want_access = pre['survivor_access'] + pre['absorbed_access']
    dims['access_count'] = {
        'ok': after_access == want_access,
        'detail': 'survivor=%d + absorbed=%d → want %d, got %d' % (
            pre['survivor_access'], pre['absorbed_access'], want_access, after_access),
    }

    after_kv = _non_sys_kv(brain, survivor_id)

    # 3. kv_fill — keys absorbed carried that survivor lacked, not dropped, land.
    fillable = {k: v for k, v in pre['absorbed_kv'].items()
                if (v or '').strip() and k not in drop and k not in over
                and not (pre['survivor_kv'].get(k) or '').strip()}
    missing = [k for k in fillable if not (after_kv.get(k) or '').strip()]
    dims['kv_fill'] = {
        'ok': not missing,
        'detail': 'fillable=%s, missing=%s' % (sorted(fillable) or 'none', missing or 'none'),
    }

    # 4. kv_survivor_wins — survivor's own non-empty values untouched (unless overridden).
    clobbered = [k for k, v in pre['survivor_kv'].items()
                 if (v or '').strip() and k not in over and after_kv.get(k) != v]
    dims['kv_survivor_wins'] = {
        'ok': not clobbered,
        'detail': 'clobbered=%s' % (clobbered or 'none'),
    }

    # 5. content — survivor's content preserved unless overridden.
    if 'content' in over:
        dims['content'] = {'ok': True, 'detail': 'overridden by caller'}
    else:
        dims['content'] = {
            'ok': _content(brain, survivor_id) == pre['survivor_content'],
            'detail': 'preserved' if _content(brain, survivor_id) == pre['survivor_content']
                      else 'CHANGED without override',
        }

    # 6 + 8. edges migrated + direction fidelity.
    after_edges = _external_edges(brain, survivor_id, absorbed_id)
    after_pairs = {(n, r) for n, r, _ in after_edges}
    after_dir = {(n, r): d for n, r, d in after_edges}
    want_edges = [(n, r, d) for n, r, d in pre['absorbed_edges'] if r not in prune]
    missing_edges = [(n, r) for n, r, _ in want_edges if (n, r) not in after_pairs]
    reversed_edges = [(n, r) for n, r, d in want_edges
                      if (n, r) in after_dir and after_dir[(n, r)] != d]
    dims['edges_migrated'] = {
        'ok': not missing_edges,
        'detail': 'want %d, missing=%s' % (len(want_edges), missing_edges or 'none'),
    }
    dims['edge_direction_fidelity'] = {
        'ok': not reversed_edges,
        'diagnostic': True,   # migration regression guard — never flips lossless
        'detail': 'migration flipped stored direction on: %s' % (
            reversed_edges or 'none'),
    }

    # 7. archived with provenance pointing at survivor.
    row = brain.conn.execute(
        "SELECT archived FROM nodes WHERE id = ?", (absorbed_id,)).fetchone()
    prov = brain._meta_kv.get_all_bulk([absorbed_id]).get(absorbed_id, {})
    archived_ok = (row and row[0] == 1
                   and prov.get('_sys_archived_survivor_id') == survivor_id)
    dims['archived_provenance'] = {
        'ok': bool(archived_ok),
        'detail': 'archived=%s survivor_ref=%s' % (
            row[0] if row else '?', prov.get('_sys_archived_survivor_id')),
    }

    hard = [v['ok'] for v in dims.values() if not v.get('diagnostic')]
    warnings = [k for k, v in dims.items() if v.get('diagnostic') and not v['ok']]
    return {'dimensions': dims, 'lossless': all(hard), 'warnings': warnings}


# ──────────────────────────────────────────────────────────────────
# Rich fixture — one absorb exercising every dimension at once.
# ──────────────────────────────────────────────────────────────────

def build_rich_fixture(brain):
    """Build survivor + absorbed + neighbors. absorbed carries refs, voice
    quotes, an emergent KV field, access, and external edges.

    Returns (survivor_id, absorbed_id).
    """
    def node(title, **kw):
        return brain.remember(type='fact', title=title,
                              encoding_source='anchor', **kw)['id']

    survivor = node('survivor — canonical', content='survivor content',
                    situation='when surviving', source_refs=['aaaaaaaa'])
    absorbed = node('absorbed — redundant', content='absorbed content',
                    situation='when absorbed (survivor wins this)',
                    source_refs=['bbbbbbbb', 'cccccccc'],
                    their_raw_quote='the operator said this',
                    my_raw_quote='Anchor reflected this')
    # Emergent KV the survivor lacks → must fill.
    brain._meta_kv.set_many(absorbed, {'emergent_key': 'emergent value'})
    # Access history.
    brain.conn.execute("UPDATE nodes SET access_count = 40 WHERE id = ?", (survivor,))
    brain.conn.execute("UPDATE nodes SET access_count = 13 WHERE id = ?", (absorbed,))
    # External edges on absorbed (should migrate).
    n_dep = node('dependency neighbor')
    n_sup = node('supporting neighbor')
    brain._graph.add_relation(absorbed, n_dep, 'depends_on',
                              description='absorbed depends on this')
    brain._graph.add_relation(n_sup, absorbed, 'supports',
                              description='this supports absorbed')
    # Intra-pair edge (should die with absorbed).
    brain._graph.add_relation(absorbed, survivor, 'similar_to')

    brain.conn.commit()
    return survivor, absorbed


def _print_report(label, report):
    print('\n=== %s ===' % label)
    for name, d in report['dimensions'].items():
        tag = 'WARN' if d.get('diagnostic') and not d['ok'] else ('PASS' if d['ok'] else 'FAIL')
        print('  [%-4s] %-26s %s' % (tag, name, d['detail']))
    print('  lossless=%s  warnings=%s' % (report['lossless'], report['warnings']))


def main():
    import tempfile
    import shutil
    from servers.brain import Brain

    tmp = tempfile.mkdtemp(prefix='absorb_probe_')
    rc = 0
    try:
        brain = Brain(os.path.join(tmp, 'brain.db'), skip_embedder=True)

        # The gate: one absorb on a richly-populated fixture must be lossless
        # across every additive dimension at once.
        survivor, absorbed = build_rich_fixture(brain)
        pre = snapshot_pre(brain, survivor, absorbed)
        brain.absorb(survivor, absorbed,
                     content='merged synthesis', reason='preservation probe')
        rep = audit(brain, survivor, absorbed, pre, overrides=['content'])
        _print_report('Rich fixture (gate — must be lossless)', rep)
        if not rep['lossless']:
            rc = 1

        brain.close()
        print('\nGate: %s' % ('PASS' if rc == 0 else 'FAIL'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
