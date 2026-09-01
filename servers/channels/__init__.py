"""Channels — the correspondents that ADDRESS a live stream.

A channel is a CORRESPONDENT plus a WIRE, indexed the way
`docs/LATERAL-SCALES.md` indexes laterals — by who is on the other end:

    self_channel/   another stream of me (`peer:` once a persistent one exists)
    thalamus/       the brain itself, speaking to its streams

"Addresses" is the boundary, not "reaches". Recall, the Frame and the boot
render all reach a live stream and live elsewhere (`scales/s1`,
`brain_voice.py`); they are output, not correspondence. Nor is this the
`additionalContext` sense of "channel" used in CLAUDE.md's Conventions — that
names the injection pipe, this names who is speaking.

NOT under `scales/`: neither package runs an `integrate(O, K) → Δ` loop.
`scales/` is the GRAIN axis — s1, s2, and the shared machinery serving them.
Both clock contracts (`test_clock_contract_sync`, `test_time_window_contract`)
can therefore name `servers/scales` as a PREFIX, so a future grain is guarded
without anyone remembering to list it. That prefix stays honest only while
nothing wall-clock-by-design sits under `scales/` — the alternative, an
`EXEMPT_FILES` entry, is what those tests explicitly ask you not to grow.
Delivery windows here are real-elapsed: the courier-class exemption documented
in `self_contract` and inherited by `thalamus_contract`, not an oversight.

A delivery is traced as an s0 K event (`self_message`, `thalamus_delivery`),
so a channel rides the S0 loop as an incoming correspondent rather than
introducing a grain of its own. Caveat while Step 8 is open: only the STOP leg
writes that trace. The boot leg delivers untraced, so
`query_traces(ref_type='thalamus_delivery')` is not yet the complete record.

`world` is NOT a channel — it is ingestion, a source you read rather than a
correspondent you address, and must not ride this transport
(`docs/LATERAL-SCALES.md`, which owns that call).

Taxonomy: docs/LATERAL-SCALES.md
"""
