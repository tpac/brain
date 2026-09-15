"""Contract test for the shared encoder prompt-closers — the edge-aspect block,
the residue review block, and the run closure — and how they assemble. The
texts are single-sourced (servers.aspects + servers.trace_contract); the
assembly is the journal component (`_inject_edge_aspects` → JournalBinding
.decorate_system). This pins shape + ordering so a future edit to the closure
wording (e.g. dropping the literal DONE the runner-loop convention expects, or
breaking the `## Review` reference) fails loudly in CI rather than silently
shipping a malformed prompt.

Deterministic, no brain/embedder/LLM — a SimpleNamespace stub feeds the aspect
dict, so this runs in CI where the real eval (journal_probe.py,
needs Sonnet) cannot.
"""
from types import SimpleNamespace as NS

from servers.aspects import render_edge_aspects_block
from servers.trace_contract import (
    render_prompt_closure, JOURNAL_INSTRUCTION)
from servers.scales.s2.base import IntegrationUnit


# ── the three render sources ──

def test_journal_strategy_is_separate_from_tool_mechanics():
    from servers.trace_contract import journal_tool_schema
    assert '`journal`' in JOURNAL_INSTRUCTION
    assert 'DONE' not in JOURNAL_INSTRUCTION
    assert '8,000' in journal_tool_schema()['description']
    assert '## Review' not in JOURNAL_INSTRUCTION


def test_closure_shape():
    c = render_prompt_closure()
    assert c.startswith('## Finishing')
    assert 'only `journal` ends the run' in c
    assert 'no tool call' in c
    assert c.rstrip().endswith('"DONE".')


def test_edge_aspects_skip_and_heading():
    # Visibility is the per-aspect `prompt_visible` fact (aspects_v1.json,
    # Step 4) — the render skips prompt-invisible and node-only aspects.
    fake = {
        'correction_improvement': NS(edge_relations=('corrects', 'supersedes'),
                                     prompt_visible=True),
        'survivor_lineage':       NS(edge_relations=('absorbed_into',),
                                     prompt_visible=False),  # skipped
        'noise':                  NS(edge_relations=('co_accessed',),
                                     prompt_visible=False),  # skipped
        'generic_relation':       NS(edge_relations=('related_to',),
                                     prompt_visible=False),  # skipped
        'identity_bearing':       NS(edge_relations=(),
                                     prompt_visible=True),   # node-only → skipped
    }
    b = render_edge_aspects_block(fake)
    assert '## Edge Aspects' in b and '## Edge Families' not in b   # renamed
    assert 'correction_improvement' in b
    assert 'survivor_lineage' not in b and 'absorbed_into' not in b
    assert 'noise' not in b and 'generic_relation' not in b
    assert 'identity_bearing' not in b
    assert 'Avoid `related_to`' in b


def test_seed_prompt_invisible_set_is_pinned():
    # The seed's prompt_visible facts reproduce the deleted
    # EDGE_ASPECT_PROMPT_SKIP contract: the two catch-alls + the system
    # aspects stay out of encoder vocabulary blocks. settlement's verbs
    # already appear via their semantic home aspects. Keep in sync with
    # tests/test_aspects_contract.py::test_prompt_invisible_set_is_pinned.
    import json
    import os
    seed_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'servers', 'scales', 's2', 'aspects_v1.json')
    with open(seed_path) as f:
        seed = json.load(f)
    invisible = {n for n, spec in seed.items() if not spec['prompt_visible']}
    assert invisible == {
        'generic_relation', 'noise', 'survivor_lineage', 'settlement'}


def test_edge_aspects_empty_when_nothing_to_show():
    assert render_edge_aspects_block({}) == ''
    assert render_edge_aspects_block(
        {'noise': NS(edge_relations=('co_accessed',), prompt_visible=False)}) == ''


# ── the assembly (the ordering contract) ──

class _Stub(IntegrationUnit):
    """Bypasses IntegrationUnit.__init__ — the closers only read brain.aspects."""
    NAME = 'consolidation'

    def __init__(self, aspects):
        self.brain = NS(aspects=NS(all=lambda: aspects))

    def run(self):  # abstract on the base
        pass


def test_assembly_order_and_done_last():
    fake = {'correction_improvement': NS(edge_relations=('corrects', 'supersedes'),
                                         prompt_visible=True)}
    s = _Stub(fake)
    body = 'BODY...\n\n## Speed\n\nbe decisive.'
    asm = s.journal.decorate_system(s._inject_edge_aspects(body))

    # all three present, in order: edge aspects → review → closure
    i_edge = asm.index('## Edge Aspects')
    i_review = asm.index(JOURNAL_INSTRUCTION)
    i_closure = asm.index('## Finishing')
    assert i_edge < i_review < i_closure
    # closure is genuinely last — DONE is the final content
    assert asm.rstrip().endswith('"DONE".')
    # body preserved ahead of the closers
    assert asm.index('## Speed') < i_edge


def test_single_shot_decoration_has_no_closure():
    """multi_round=False (a single-shot request) gets the review block but NO
    closure — there is no terminal-turn ambiguity to disambiguate, and a DONE
    instruction on a one-turn call is noise."""
    s = _Stub({})
    asm = s.journal.decorate_system('BODY...', multi_round=False)
    assert JOURNAL_INSTRUCTION in asm
    assert '## Finishing' not in asm and 'DONE' not in asm


def test_arc_binding_orders_arc_before_review():
    """An arc-bound binding (the S1 Scribe) appends `## Arc` before the review
    block — §7.2 order, pinned since a fenceless Arc reaching forward into the
    Review fence is the silent-corruption case the scanner guards."""
    from servers.scales.journal import JournalBinding
    b = JournalBinding(None, scale='s1', session_id='sess', arc=True)
    asm = b.decorate_system('BODY...')
    assert asm.index('## Arc') < asm.index(JOURNAL_INSTRUCTION) < asm.index('## Finishing')


def test_inject_edge_aspects_noop_when_empty():
    """No-op append (just rstrip) when there are no edge aspects to show."""
    s = _Stub({})
    assert s._inject_edge_aspects('BODY...') == 'BODY...'
