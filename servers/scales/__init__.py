"""Scales — the GRAIN axis of the fractal.

What belongs here: a unit that runs an `integrate(O, K) → Δ` loop (`s1/`,
`s2/`), or the shared machinery serving those units (`dispatch.py`,
`journal.py`, `runner.py` — these run no loop themselves, but exist only for
the ones that do).

What does NOT: anything whose clock is real-elapsed wall-clock. Both clock
contracts (`tests/test_clock_contract_sync.py`,
`tests/test_time_window_contract.py`) scan this tree by PREFIX, so every
future grain is guarded without anyone remembering to list it — and that only
holds while nothing here needs an exemption. A package that ADDRESSES a live
stream (real-elapsed delivery windows, roster staleness) is a correspondent,
not a grain: it belongs in `servers/channels/`, the way `self_channel` and
`thalamus` do. Moving it is the fix; growing `LEGITIMATE_USES` is not.

Shape and rationale: docs/ARCHITECTURE-FRACTAL.md
"""
