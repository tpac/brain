"""Render replay journal ages on the conversation clock, retaining real storage.

The real JournalBinding still harvests, resolves, pins and counts notes. Only
its displayed first_seen date is mapped from a harvest timestamp to the
fixture window that produced it. Transaction timestamps are never rewritten.
"""
from datetime import datetime


class ReplayJournal:
    def __init__(self, binding):
        if binding.addressed:
            raise ValueError('This replay adapter requires an unaddressed eval journal')
        self.binding = binding
        self._window_now = None
        self._dates = {}

    def __getattr__(self, name):
        return getattr(self.binding, name)

    def set_window(self, now):
        self._window_now = datetime.strptime(now, '%Y-%m-%d %H:%M UTC').strftime('%Y-%m-%dT%H:%M:00+00:00')

    def _notes(self):
        return self.binding.brain.journal_notes(
            scale=self.binding.scale, unit=self.binding.unit,
            session_id=self.binding.session_id)

    def harvest(self, final_text, chain_id, arc_limit=800):
        if self._window_now is None:
            raise ValueError('Set the fixture window before harvesting')
        result = self.binding.harvest(final_text, chain_id, arc_limit=arc_limit)
        for note in self._notes():
            if note['chain_id'] == chain_id:
                stamp = note['created_at']
                previous = self._dates.setdefault(stamp, self._window_now)
                if previous != self._window_now:
                    raise ValueError('Two fixture windows share a journal timestamp')
        return result

    def continuity(self):
        from servers.trace_contract import render_journal_notes_prefix
        notes = self._notes()
        rendered = []
        for note in notes:
            row = dict(note)
            if row.get('first_seen'):
                # Unknown provenance must fail loudly rather than present a
                # wall-clock age as a conversation date.
                row['first_seen'] = self._dates[row['first_seen']]
            rendered.append(row)
        return render_journal_notes_prefix(rendered)
