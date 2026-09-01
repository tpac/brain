"""Channels — everything that reaches a live stream.

A channel is a CORRESPONDENT plus a WIRE. Indexed the way
`docs/LATERAL-SCALES.md` indexes laterals — by who is on the other end:

    self_channel/   another stream of me (`peer:` once a persistent one exists)
    thalamus/       the brain itself, speaking to its streams

NOT under `scales/`, and the distinction is load-bearing rather than tidy:
neither package runs an `integrate(O, K) → Δ` loop. Both ride the S0 loop as
incoming correspondents, traced as s0 K events (`self_message`,
`thalamus_delivery`). `scales/` is the GRAIN axis — s1, s2, and the machinery
serving them — so the conversation-time contract that governs it
(`tests/test_clock_contract_sync.py`) can name `servers/scales` as a prefix
only because channels live outside it. Delivery windows here are real-elapsed
wall-clock: the documented courier-class exemption, not an oversight.

`world` does NOT belong here. It is ingestion — reading a source, not
addressing a correspondent — and LATERAL-SCALES is explicit that it must not
be jammed onto the message bus. Zero shared transport.

Taxonomy: docs/LATERAL-SCALES.md
"""
