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

    `source` is the producer's encoding_source. A binding WITH a source is
    ADDRESSED: its review lines tagged tell/ask are messages to the people
    working, filed through the Thalamus door instead of written as residue.
    The audience falls out of the binding: a session (the Scribe) makes the
    item directed — a tell or an ask reaches that session at its next Stop;
    no session (an S2 unit) makes it the door's default audience — a tell
    reaches the first session to pull at boot or Stop, an ask reaches every
    session at boot until answered. One instruction, one door, two
    audiences. Without a source (a bare eval binding) they stay plain notes.
    """

    def __init__(self, brain, *, scale, unit='', session_id='', arc=False,
                 source=''):
        self.brain = brain
        self.scale = scale
        self.unit = unit
        self.session_id = session_id
        self.arc = arc
        self.source = source
        self.addressed = bool(source)

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
        res = {}
        try:
            res = self.brain.write_journal_notes(
                final_text=final_text, chain_id=chain_id,
                scale=self.scale, session_id=self.session_id)
        except Exception as e:
            self.brain._log_error(
                self._log_key('write'), e,
                'residue note write failed — run otherwise intact')
        # The routing half is isolated like the write half: a Thalamus read
        # or write hiccup must never abort the encoder's run (S2's batch loop
        # relies on it) nor cost the session arc below.
        try:
            if self.addressed:
                self._route_addressed(res, chain_id)
            elif res.get('addressed'):
                # No source to file under — the lines stay residue, but a
                # message nobody will deliver is worth a warning.
                self.brain._log_warning(
                    self._log_key('addressed_unbound'),
                    'chain=%s: %d tell/ask line(s) from a binding without a '
                    'source — written as plain notes, undelivered'
                    % (chain_id, len(res['addressed'])))
                self._write_rows(res['addressed'], chain_id)
        except Exception as e:
            self.brain._log_error(
                self._log_key('route'), e,
                'addressed-line routing failed — run otherwise intact')
        if self.arc:
            self.brain.write_session_arc(
                final_text=final_text, session_id=self.session_id,
                limit=arc_limit)
        return strip_journal_sections(final_text)

    def _route_addressed(self, res, chain_id):
        """File this run's tell/ask lines as Thalamus items — the non-LLM
        entrance to the one door: `tell` is a notice, `ask` needs an answer;
        the audience is the binding's (see the class docstring). The
        normalized subject is the item's dedup_key — "one line per subject"
        holds literally: a re-assertion updates, never duplicates — and a
        node-id subject is also its ref. The run chain rides along so the
        filing is traced on it. A door rejection is LOUD and the line is
        kept as a plain note carrying the reason, so the residue survives
        and the encoder reads the rejection next run. A `resolved · <subject>`
        line also withdraws this binding's open item under that key — the
        encoder's existing verb closes its own item."""
        addressed = res.get('addressed') or []
        resolved = res.get('resolved') or []
        if not (addressed or resolved):
            return
        from servers.channels.thalamus import thalamus
        from servers.trace_contract import (JOURNAL_ASK_TAG, journal_key,
                                            journal_subject_refs,
                                            resolve_target)
        kept = []
        for n in addressed:
            try:
                r = thalamus.file(
                    self.brain, self.source, n['note'],
                    needs_answer=journal_key(n.get('tag')) == JOURNAL_ASK_TAG,
                    for_whom=self.session_id or None,
                    dedup_key=journal_key(n['subject']),
                    refs=journal_subject_refs(n['subject']) or None,
                    session_id=self.session_id, run_chain=chain_id)
            except Exception as e:
                r = {'ok': False, 'error': str(e)}
            if not r.get('ok'):
                self.brain._log_warning(
                    self._log_key('addressed_rejected'),
                    'chain=%s: %s · %s not filed — %s' % (
                        chain_id, n.get('tag'), n['subject'], r.get('error')))
                kept.append(dict(n, undelivered=r.get('error') or 'rejected'))
        if kept:
            self._write_rows(kept, chain_id)
        if resolved:
            # Close only what this binding actually has open (≤ the budget
            # cap, one read) — with the read side's tolerance for an echoed
            # `tag · subject · …` head (resolve_target), so the same resolve
            # line that retires the note also closes the item.
            open_keys = {i['dedup_key'] for i in thalamus.list_items(
                             self.brain, source=self.source,
                             target_session=self.session_id)['items']
                         if i.get('dedup_key')
                         and i.get('target_session', '') == self.session_id}
            for r in resolved:
                target = resolve_target(journal_key(r['subject']), r['note'],
                                        open_keys)
                if target in open_keys:
                    thalamus.withdraw(self.brain, self.source,
                                      dedup_key=target,
                                      target_session=self.session_id)

    def _write_rows(self, notes, chain_id):
        self.brain.write_journal_note_rows(
            notes, chain_id=chain_id, scale=self.scale,
            session_id=self.session_id)
