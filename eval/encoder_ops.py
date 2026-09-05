"""Op-dump reading for the encoder A/B tools — ONE reader for the ops an
encoder run emits, shared by eval/encoder_prompt_ab.py (behavior + gold
scoring) and eval/encoder_ops_shape.py (shape metrics), so the two never count
the same dump differently (docs/REVISE-SHAPE-SPEC.md §8 row 10).

Swap-aware: on revise a field is its new value or `{old, new}` swaps
(contract.REVISE_RULE); every text this module returns is the NEW text. The
removed `old` is precisely what a correct swap deletes — a fact check that
read it would score the stale value as still carried.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.contract import (connect_to_target, is_swap, is_swap_list,  # noqa: E402
                              unwrap_operations)

# The node surfaces a revise can write besides content — the field-coverage
# instrument reads which of them an op touched.
SURFACE_FIELDS = ('title', 'situation', 'question', 'reasoning', 'thought',
                  'type', 'evolution_status')


def ops_of(write):
    """The individual ops one recorded write carries. `write` is the harness
    log entry {'tool', 'args'}. Batch tools carry a list under one key:
    `operations` (brain_batch — may arrive JSON-encoded; unwrapped the way
    dispatch does), `nodes` (remember_batch — its batch-level `connect_to`
    applies the same edge from EVERY created node, so it is folded into each
    node's own list, on copies), `revisions`, `connections`. A bare args dict
    is one op; non-dict items are dropped."""
    args = write.get('args') or {}
    ops = args.get('operations')
    if isinstance(ops, str):
        ops = unwrap_operations(ops)
    if isinstance(ops, list):
        return [o for o in ops if isinstance(o, dict)]
    nodes = args.get('nodes')
    if isinstance(nodes, list):
        shared = [e for e in (args.get('connect_to') or []) if isinstance(e, dict)]
        out = []
        for n in nodes:
            if isinstance(n, dict):
                if shared:
                    n = dict(n, connect_to=list(n.get('connect_to') or []) + shared)
                out.append(n)
        return out
    for key in ('revisions', 'connections'):
        v = args.get(key)
        if isinstance(v, list):
            return [o for o in v if isinstance(o, dict)]
    return [args] if args else []


def kind(op):
    """The op's declared `op`, else inferred from its keys — single-purpose
    batches carry no `op`: node_id → revise, source_id+target_id → connect,
    otherwise remember."""
    k = op.get('op')
    if k:
        return k
    if op.get('node_id'):
        return 'revise'
    if op.get('source_id') and op.get('target_id'):
        return 'connect'
    return 'remember'


def new_text(value):
    """The text a value PROPOSES: a bare string as is; a swap or a swap list →
    its `new` strings, joined; a list of scalars (source_refs) joined; None →
    ''."""
    if value is None:
        return ''
    if is_swap(value):
        return str(value.get('new') or '')
    if is_swap_list(value):
        return ' '.join(str(e.get('new') or '') for e in value)
    if isinstance(value, (list, tuple)):
        return ' '.join(str(x) for x in value)
    return str(value)


def swaps_of(op):
    """{field: [swap, ...]} for every field a revise patches — the
    content_edits alias normalized onto `content`. Bare values are not swaps
    and are absent here."""
    out = {}
    for field, value in op.items():
        if field == 'content_edits':
            field = 'content'
        if is_swap(value):
            out.setdefault(field, []).append(value)
        elif is_swap_list(value):
            out.setdefault(field, []).extend(value)
    return out


def edge_entries(op):
    """Every edge an op asserts, normalized to
    {via, source, target, relation, why}: `connect_to` on a remember (source
    None — the node has no id yet) or on a revise/absorb (source = the node's
    id), the `relations: [{relation, why}]` form one entry per relation, and a
    standalone connect (`description` is its why). A `disconnect` removes an
    edge and asserts none — it yields no entry, so it can never stand in for
    (or blank out) the pair's real why in a scorer. Relation and why are new
    text (swap-aware); the target is read through contract.connect_to_target,
    so the deprecated `title` key still counts."""
    out = []
    k = kind(op)
    src = (str(op.get('node_id') or op.get('survivor_id') or '')[:8]
           if k in ('revise', 'absorb') else None)
    for c in (op.get('connect_to') or []):
        if not isinstance(c, dict):
            continue
        target = str(connect_to_target(c) or '')
        rels = [r for r in (c.get('relations') or []) if isinstance(r, dict)] or [c]
        for r in rels:
            out.append({'via': 'connect_to', 'source': src or None,
                        'target': target,
                        'relation': new_text(r.get('relation')),
                        'why': new_text(r.get('why'))})
    if k == 'connect':
        out.append({'via': 'connect',
                    'source': str(op.get('source_id') or '')[:8],
                    'target': str(op.get('target_id') or ''),
                    'relation': new_text(op.get('relation')),
                    'why': new_text(op.get('description') or op.get('why'))})
    return out


def revise_surfaces(op):
    """Which node SURFACES a revise WROTE, with the new text of each — the
    field-coverage question (id:450650d5): a stale value lives in several
    separately-embedded surfaces, and a revise that fixes title+content while
    leaving the same value in `situation` reads as a pass under any text-only
    check. Content counts whether whole, as swaps, or via the content_edits
    alias."""
    out = {f: new_text(op.get(f)) for f in SURFACE_FIELDS if op.get(f)}
    body = ' '.join(t for t in (new_text(op.get('content')),
                                new_text(op.get('content_edits'))) if t)
    if body:
        out['content'] = body
    return out


def revise_text(op):
    """The prose a revise PROPOSES — content, situation, title as new text;
    the removed `old` never satisfies a fact check."""
    surfaces = revise_surfaces(op)
    return ' '.join(surfaces.get(f, '') for f in ('content', 'situation', 'title'))
