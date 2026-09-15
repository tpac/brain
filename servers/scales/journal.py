"""One scoped journal binding for every encoder.

The trace contract owns the JSON state, native tool schema, strategy text
and Arc format. The binding owns invocation selection, shown-version receipts,
scoped tool dispatch and live routing. Provider objects stay in the runner.

S1 and S2 retain distinct purpose text. Loop callers bind tools and dispatch
at their runner seam; single-shot callers execute the same journal tool from
one response. Arc is independent session context harvested from final prose.
"""


class JournalBinding:
    """One encoder's attachment to the journal.

    Identity mirrors journal_view() scoping: S2 units bind (scale, unit);
    the S1 Scribe binds (scale, session_id) — its residue is session-walled.
    `arc=True` opts into the second journal object (the `## Arc` fence →
    session arc accumulator); only the Scribe carries it today.

    `source` is the producer's encoding_source. A binding WITH a source is
    ADDRESSED: its tell/ask operations are messages to the people
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
        self._run_chain = None
        self._view = None
        self._selection_failed = False
        self._reference_context = None
        self.stats = {}

    # Stable log keys let operational queries span journal versions.
    def _log_key(self, op):
        if self.scale == 's1':
            return 's1e_journal_notes_%s' % op
        return 's2_%s_journal_%s' % (self.unit, op)

    def continuity(self, *, chain_id=None):
        """Prepare current continuity before each independent request.

        A chain identifies an invocation, not a batch. Freeze its private
        selection, refresh committed lifecycle state and message outcomes.
        A new chain resets the selection; omission of chain_id is the
        single-request convenience used by standalone readers.
        """
        from servers.trace_contract import render_journal_view
        if chain_id is None or chain_id != self._run_chain:
            self._run_chain, self._view = chain_id, None
            self._selection_failed = False
        try:
            if not self._selection_failed:
                self._view = self.brain.journal_view(
                    scale=self.scale, unit=self.unit, session_id=self.session_id,
                    previous=self._view, invocation=chain_id)
            view = self._view or {'selection_failed': True}
        except Exception as e:
            # No initial snapshot means there is no safe private selection
            # to recover mid-run: a later load would echo this run's notes.
            # Retry on the next invocation; live messages remain independent.
            self._selection_failed = self._view is None
            self.brain._log_error(
                self._log_key('read'), e,
                'journal continuity read failed — retaining last known state')
            view = dict(self._view or {}, stale=True,
                        selection_failed=self._selection_failed)
        try:
            text, stats, references = render_journal_view(view)
        except Exception as e:
            self.brain._log_error(self._log_key('render'), e,
                                  'journal rendering failed — encoding without residue')
            text, stats, references = render_journal_view({'render_failed': True})
        self._reference_context = {
            'chain_id': chain_id, 'view': self._view,
            'references': references,
        }
        messages = self.messages()
        self.stats = dict(stats, message_chars=len(messages),
                          continuity_chars=len(text) + len(messages))
        try:
            self.brain.log_debug('journal_continuity', 'JournalBinding',
                                 chain_id=chain_id or '',
                                 cursor=view.get('cursor', 0), **self.stats)
        except Exception as e:
            self.brain._log_error(self._log_key('telemetry'), e,
                                  'journal continuity telemetry failed')
        return text + messages

    def messages(self):
        """Live outcomes, read for EACH request by continuity().

        Earlier batches can file or resolve an item. Reusing their input
        snapshot lets a later batch overwrite the newer message or miss an
        answer. Thalamus remains the owner of delivery state.
        """
        if not self.addressed:
            return ''
        try:
            return self._producer_view()
        except Exception as e:
            self.brain._log_error(
                self._log_key('view'), e,
                'producer view read failed — encoding without it')
            return ''

    def _producer_view(self):
        """The join for the producer view: this binding's own items (the
        door orders and windows them — producer_items) as rows the contract
        phrases. The tag/subject mapping is the exact inverse of the forward
        map in _route_addressed (tag → needs_answer, subject → dedup_key). A
        refused filing already sits in the residue notes with its reason."""
        from servers.channels.thalamus import thalamus
        from servers.channels.thalamus import thalamus_contract as tc
        from servers.trace_contract import (render_producer_view,
                                            JOURNAL_TELL_TAG, JOURNAL_ASK_TAG,
                                            JOURNAL_RUN_SUBJECT)
        rows, seen = [], set()
        for i in thalamus.producer_items(
                self.brain, self.source, self.session_id,
                settled_days=tc.PRODUCER_VIEW_SETTLED_DAYS):
            # One row per subject: the door orders open-first, so a key
            # re-asked after a settlement shows its live state, not both.
            if i['dedup_key'] in seen:
                continue
            seen.add(i['dedup_key'])
            rows.append({
                'tag': (JOURNAL_ASK_TAG if tc.kind_of(i) == tc.KIND_ASK
                        else JOURNAL_TELL_TAG),
                'subject': i['dedup_key'] or JOURNAL_RUN_SUBJECT,
                'note': i['body'], 'fate': tc.fate_of(i), 'answer': i['answer'],
            })
        return render_producer_view(rows)

    def decorate_system(self, system_prompt, multi_round=True):
        """The WRITE-side instructions, appended at the system tail in the
        contract order: arc (when bound) → review block → closure. The
        closure (terminal-turn definition + `## Review` placement + DONE)
        applies only to multi-round requests and must be genuinely last —
        call this AFTER all other prompt assembly (e.g. edge-aspect vocab).
        """
        from servers.trace_contract import (render_journal_arc_block,
                                            JOURNAL_INSTRUCTION,
                                            render_prompt_closure)
        from servers.trace_contract import JOURNAL_S1_INSTRUCTION, JOURNAL_S2_INSTRUCTION
        purpose = JOURNAL_S1_INSTRUCTION if self.scale == 's1' else JOURNAL_S2_INSTRUCTION
        out = system_prompt.rstrip() + '\n\n' + purpose
        if self.arc:
            out = out.rstrip() + '\n\n' + render_journal_arc_block()
        out = out.rstrip() + '\n\n' + JOURNAL_INSTRUCTION
        if multi_round:
            out = out.rstrip() + '\n\n' + render_prompt_closure()
        return out

    def bind_tools(self, tools, dispatch_fn, chain_id):
        """Attach the scoped journal tool to a generic encoder loop."""
        from servers.trace_contract import journal_tool_schema, JOURNAL_TOOL_NAME

        def dispatch(name, arguments):
            if name == JOURNAL_TOOL_NAME:
                return self.apply(arguments, chain_id)
            return dispatch_fn(name, arguments)

        return dict(tools=[*tools, journal_tool_schema()], dispatch_fn=dispatch,
                    terminal_tools=(JOURNAL_TOOL_NAME,))

    def apply(self, arguments, chain_id):
        """Execute a native journal call, preserving independent live operations."""
        context, self._reference_context = self._reference_context, None
        if not isinstance(arguments, dict) or set(arguments) != {'operations'}:
            message = 'journal requires only an operations array'
            self.brain._log_warning(self._log_key('arguments'), message)
            return {'ok': False, 'error': message}
        try:
            res = self.brain.write_journal_operations(
                arguments['operations'], chain_id=chain_id,
                scale=self.scale, session_id=self.session_id,
                unit=self.unit, context=context)
        except Exception as e:
            self.brain._log_error(self._log_key('write'), e,
                                  'journal tool failed — task work preserved')
            return {'ok': False, 'error': str(e)}
        try:
            if self.addressed:
                self._route_addressed(res, chain_id)
            elif res.get('addressed'):
                self.brain._log_warning(
                    self._log_key('addressed_unbound'),
                    'No source for live messages; retained as undelivered notes')
                self._write_rows([dict(n, undelivered='no bound producer')
                                  for n in res['addressed']], chain_id)
        except Exception as e:
            self.brain._log_error(self._log_key('route'), e,
                                  'journal routing failed — task work preserved')
            return {'ok': False, 'error': str(e), 'result': res}
        return {'ok': res['status'] == 'ok', 'result': res,
                **({'error': 'Some journal operations were rejected'} if res['status'] != 'ok' else {})}

    def harvest_arc(self, final_text, arc_limit=800):
        """Arc is separate session context; journal writes are native tool calls."""
        if self.arc:
            self.brain.write_session_arc(final_text=final_text, session_id=self.session_id,
                                         limit=arc_limit)

    def _route_addressed(self, res, chain_id):
        """File this run's tell/ask lines as Thalamus items — the non-LLM
        entrance to the one door: `tell` is a notice, `ask` needs an answer;
        the audience is the binding's (see the class docstring). The
        normalized subject is the item's dedup_key — "one line per subject"
        holds literally: a re-assertion updates, never duplicates — and a
        node-id subject is also its ref. The run chain rides along so the
        filing is traced on it. A door rejection is LOUD and the line is
        kept as a plain note carrying the reason. Only an explicit withdraw
        operation withdraws a live message; private persistence is independent."""
        addressed = res.get('addressed') or []
        withdrawn = res.get('withdrawn') or []
        if not (addressed or withdrawn):
            return
        from servers.channels.thalamus import thalamus
        from servers.trace_contract import (JOURNAL_ASK_TAG, journal_key,
                                            journal_subject_refs)
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
        if withdrawn:
            # Only this producer's messages in the bound audience are eligible.
            open_keys = {i['dedup_key'] for i in thalamus.producer_items(
                             self.brain, self.source, self.session_id)
                         if i['dedup_key']}
            for r in withdrawn:
                target = journal_key(r['subject'])
                if target in open_keys:
                    try:
                        outcome = thalamus.withdraw(
                            self.brain, self.source, dedup_key=target,
                            target_session=self.session_id)
                    except Exception as exc:
                        outcome = {'ok': False, 'error': str(exc)}
                    if not outcome.get('ok'):
                        reason = outcome.get('error') or 'withdrawal rejected'
                        self.brain._log_warning(self._log_key('withdraw_failed'), reason)
                        kept.append(dict(
                            tag='failure', subject=target,
                            note='Message withdrawal failed: ' + reason))
        # Feedback storage must not prevent independent live operations.
        if kept:
            self._write_rows(kept, chain_id)

    def _write_rows(self, notes, chain_id):
        self.brain.write_journal_note_rows(
            notes, chain_id=chain_id, scale=self.scale,
            session_id=self.session_id, unit=self.unit)
