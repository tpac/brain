"""Contract test for the shared encoder prompt-closers — the edge-aspect block,
the residue review block, and the run closure — and how IntegrationUnit assembles
them. These three are single-sourced (servers.aspects + servers.trace_contract)
and appended by base methods (`_inject_edge_aspects` → `_inject_review_block` →
`_append_closure`); this pins their shape + ordering so a future edit to the
closure wording (e.g. dropping the literal DONE the runner-loop convention
expects, or breaking the `## Review` reference) fails loudly in CI rather than
silently shipping a malformed prompt.

Deterministic, no brain/embedder/LLM — a SimpleNamespace stub feeds the aspect
dict, so this runs in CI where the real eval (sim_consolidation_journal.py,
needs Sonnet) cannot.
"""
from types import SimpleNamespace as NS

from servers.aspects import render_edge_aspects_block, EDGE_ASPECT_PROMPT_SKIP
from servers.trace_contract import (
    render_prompt_closure, render_journal_review_block, JOURNAL_REVIEW_INSTRUCTION)
from servers.scales.s2.base import IntegrationUnit


# ── the three render sources ──

def test_review_block_is_closure_free():
    """The review block defines the `## Review` artifact ONLY — it must carry no
    closure/terminal-turn/DONE language (Tom's decoupling: removing or relocating
    the review must never drag the closure)."""
    rb = render_journal_review_block()
    assert rb == JOURNAL_REVIEW_INSTRUCTION       # unchanged by the closure split
    assert 'DONE' not in rb
    assert 'Finishing' not in rb
    assert 'no tool call' not in rb
    assert '## Review' in rb                        # it DOES name the artifact's heading


def test_closure_shape():
    """The closure carries the terminal-turn definition (incl. the no-tool-call
    branch — the no-action-batch fix), references `## Review` by name, and ends
    with the DONE stop signal."""
    c = render_prompt_closure()
    assert c.startswith('## Finishing')
    assert 'no tool call' in c                      # terminal-turn defined as the runner does
    assert 'no tool call at all' in c               # the no-action branch is present
    assert '`## Review`' in c                       # references the artifact, by name
    assert c.rstrip().endswith('"DONE".')           # stop signal is last
    # closure must NOT redefine the note format (that's the review block's job)
    assert 'tag · subject · note' not in c


def test_edge_aspects_skip_and_heading():
    fake = {
        'correction_improvement': NS(edge_relations=('corrects', 'supersedes')),
        'survivor_lineage':       NS(edge_relations=('absorbed_into',)),   # skipped
        'noise':                  NS(edge_relations=('co_accessed',)),     # skipped
        'generic_relation':       NS(edge_relations=('related_to',)),      # skipped
        'identity_bearing':       NS(edge_relations=()),                   # node-only → skipped
    }
    b = render_edge_aspects_block(fake)
    assert '## Edge Aspects' in b and '## Edge Families' not in b   # renamed
    assert 'correction_improvement' in b
    assert 'survivor_lineage' not in b and 'absorbed_into' not in b
    assert 'noise' not in b and 'generic_relation' not in b
    assert 'identity_bearing' not in b
    assert 'Avoid `related_to`' in b
    assert EDGE_ASPECT_PROMPT_SKIP == ('generic_relation', 'noise', 'survivor_lineage')


def test_edge_aspects_empty_when_nothing_to_show():
    assert render_edge_aspects_block({}) == ''
    assert render_edge_aspects_block({'noise': NS(edge_relations=('co_accessed',))}) == ''


# ── the assembly (the ordering contract) ──

class _Stub(IntegrationUnit):
    """Bypasses IntegrationUnit.__init__ — the closers only read brain.aspects."""
    NAME = 'consolidation'

    def __init__(self, aspects):
        self.brain = NS(aspects=NS(all=lambda: aspects))

    def run(self):  # abstract on the base
        pass


def test_assembly_order_and_done_last():
    fake = {'correction_improvement': NS(edge_relations=('corrects', 'supersedes'))}
    s = _Stub(fake)
    body = 'BODY...\n\n## Speed\n\nbe decisive.'
    asm = s._append_closure(s._inject_review_block(s._inject_edge_aspects(body)))

    # all three present, in order: edge aspects → review → closure
    i_edge = asm.index('## Edge Aspects')
    i_review = asm.index('Your review')
    i_closure = asm.index('## Finishing')
    assert i_edge < i_review < i_closure
    # closure is genuinely last — DONE is the final content
    assert asm.rstrip().endswith('"DONE".')
    # body preserved ahead of the closers
    assert asm.index('## Speed') < i_edge


def test_inject_edge_aspects_noop_when_empty():
    """No-op append (just rstrip) when there are no edge aspects to show."""
    s = _Stub({})
    assert s._inject_edge_aspects('BODY...') == 'BODY...'
