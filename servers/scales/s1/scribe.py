"""S1 Scribe — the turn encoder as an in-process integration unit.

S1 Scribe is one of S1's integration units — the same O/K/Δ shape as the S2
units, run at faster cadence (every Nth conversational turn). O = the session's
turns + surface selections; K = node catalog + encoding journal + the learnable
`s1e` interaction prompt; Δ = encoded nodes/edges. So it subclasses
`IntegrationUnit` and writes through the shared in-process encoder dispatch
(`_make_encoder_dispatch`), exactly like consolidation / community / healer —
which is what makes its revise/edge traces carry the run chain (the attribution
the legacy bg-thread + TCP path silently dropped onto a date-fallback chain).

What stays S1-specific:
- **Chain format** — `s1e-{session}-{stop}` (session-scoped, turn-cadenced),
  NOT S2's time-based `s2-{ts}-{unit}`. `chain_id()` is overridden to the
  single source (SessionContext.s1e_chain).
- **The encode logic** — `encode.run_encoding(brain, dispatch, counter,
  session_id)` stays the standalone callable so the eval harnesses
  (`longmem/replay`, `s1s_ab_wiring_check`) drive it directly with their own
  brain + dispatch. `S1Scribe` is the *production* wrapper: it binds the
  daemon's brain + the in-process dispatch + the run chain.

Execution: in-process on the daemon's brain — writes serialize under
`brain.write_lock`, vectors fill via the async embed_queue. Replaces the legacy
`run_in_background` + `make_scale_dispatch` path (a throwaway
`Brain(skip_embedder=True)` copy + TCP write-back), which was S1's lone
remaining use of that pattern.
"""

from servers.scales.s2.base import IntegrationUnit


class S1Scribe(IntegrationUnit):
    """S1 turn encoder. Triggered every Nth conversational turn (and, later, on
    idle for the tail). Writes via the shared encoder dispatch so its revise /
    edge traces carry the s1e run chain."""

    NAME = 'scribe'
    SCALE = 's1'
    ENCODING_SOURCE = 'encoder:sonnet'

    O_SOURCES = ['s0_conversation', 'surface_selections']
    K_SOURCES = ['interaction:s1e', 'node_catalog', 'encoding_journal', 'session_arc']

    def __init__(self, brain, session_id, counter, dispatch_fn=None):
        super().__init__(brain, dispatch_fn)
        self.session_id = session_id
        self.counter = counter

    def chain_id(self):
        """S1's run chain — ``s1e-{session_short}-{stop}`` — from the single
        source (SessionContext.s1e_chain), NOT S2's time-based format. Cached
        per run like the base, so every write in this run shares one chain."""
        if not getattr(self, '_chain_id', None):
            from servers.session_context import SessionContext
            self._chain_id = SessionContext(
                self.session_id, stop_counter=self.counter).s1e_chain()
        return self._chain_id

    def run(self):
        """Encode this window: bind the daemon's brain + the in-process encoder
        dispatch (which stamps the s1e run chain on writes) to the standalone
        run_encoding core."""
        from servers.scales.s1.encode import run_encoding
        dispatch_fn = self._make_encoder_dispatch()
        return run_encoding(self.brain, dispatch_fn, self.counter, self.session_id)
