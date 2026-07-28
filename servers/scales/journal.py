"""The journal component — one object that attaches the encoder journal
(the residue contract) to ANY agent request, loop or single-shot.

The journal's TEXT and FORMAT live in trace_contract (single source: the
review block, the closure, the notes renderer, the `tag · subject · note`
parser, the fence scanner). This component owns the ORCHESTRATION: which
blocks decorate a request, in what order, and how residue is harvested from
the response — so an encoder binds the whole mechanism in one place instead
of hand-assembling injects.

There are no per-shape dialects — one wire format, one parser, one
instruction block. Two request-derived conditionals only:
  • the closure block is appended only for multi-round (loop) requests — it
    exists to disambiguate the terminal turn, which a single-shot call
    doesn't have;
  • harvest() returns the final text with the journal sections stripped, so
    a single-shot response's JSON payload survives extract_json (whose
    rfind-based scan would be corrupted by a `]`/`}` inside a fence that
    follows the payload). Loop callers simply ignore the return value.

Placement rules the component enforces (earned decisions, see brain nodes
2a81ea82 / 0e1a7303 / 7905e385): instruction blocks go at the SYSTEM tail —
arc (when bound) before review, closure last; the continuity notes are
CONTEXT, rendered by continuity() for the caller to place in user content
(S2 units prepend it; S1E embeds it in its structured layout — placement is
the binding's business where shipped layouts differ, the text is not).

Provider separation: this module deals only in strings and note dicts —
never SDK types. The provider seam is scales/runner.py.
"""


class JournalBinding:
    """One encoder's attachment to the journal.

    Identity mirrors journal_notes() scoping: S2 units bind (scale, unit);
    the S1 Scribe binds (scale, session_id) — its residue is session-walled.
    `arc=True` opts into the second journal object (the `## Arc` fence →
    session arc accumulator); only the Scribe carries it today.
    """

    def __init__(self, brain, *, scale, unit='', session_id='', arc=False):
        self.brain = brain
        self.scale = scale
        self.unit = unit
        self.session_id = session_id
        self.arc = arc

    # Error-log keys preserve the pre-component vocabulary so log continuity
    # survives the refactor (s1e_* for the Scribe, s2_{unit}_* for S2 units).
    def _log_key(self, op):
        if self.scale == 's1':
            return 's1e_journal_notes_%s' % op
        return 's2_%s_journal_%s' % (self.unit, op)

    def continuity(self):
        """The READ side: last K note-bearing runs' notes rendered as the
        self-labeled RECENT REVIEW NOTES block ('' when there are none — a
        clean history adds nothing). Failure-isolated: a transient logs.db
        read error must never abort an otherwise-valid encode — degrade to
        no continuity, log loud."""
        from servers.trace_contract import render_journal_notes_prefix
        try:
            notes = self.brain.journal_notes(
                scale=self.scale, unit=self.unit, session_id=self.session_id)
            return render_journal_notes_prefix(notes)
        except Exception as e:
            self.brain._log_error(
                self._log_key('read'), e,
                'residue continuity read failed — encoding without it')
            return ''

    def decorate_system(self, system_prompt, multi_round=True):
        """The WRITE-side instructions, appended at the system tail in the
        contract order: arc (when bound) → review block → closure. The
        closure (terminal-turn definition + `## Review` placement + DONE)
        applies only to multi-round requests and must be genuinely last —
        call this AFTER all other prompt assembly (e.g. edge-aspect vocab).
        """
        from servers.trace_contract import (render_journal_arc_block,
                                            render_journal_review_block,
                                            render_prompt_closure)
        out = system_prompt
        if self.arc:
            out = out.rstrip() + '\n\n' + render_journal_arc_block()
        out = out.rstrip() + '\n\n' + render_journal_review_block()
        if multi_round:
            out = out.rstrip() + '\n\n' + render_prompt_closure()
        return out

    def harvest(self, final_text, chain_id, arc_limit=800):
        """The response side: write this run's residue notes (+ the session
        arc when bound) and return `final_text` with the journal sections
        stripped — the payload remainder a single-shot caller parses JSON
        from. Note-write failures are isolated (logged loud, run intact);
        write_session_arc is failure-isolated internally.
        """
        from servers.trace_contract import strip_journal_sections
        try:
            self.brain.write_journal_notes(
                final_text=final_text, chain_id=chain_id,
                scale=self.scale, session_id=self.session_id)
        except Exception as e:
            self.brain._log_error(
                self._log_key('write'), e,
                'residue note write failed — run otherwise intact')
        if self.arc:
            self.brain.write_session_arc(
                final_text=final_text, session_id=self.session_id,
                limit=arc_limit)
        return strip_journal_sections(final_text)
