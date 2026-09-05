"""Code default for interaction `s1e_gist` — the encoder's operating rules
restated at the payload's recency position, the last instruction before
`<timeline>`. Editing SYSTEM_PROMPT here IS the deployment: every install
without a deployed override follows on the next daemon restart. Per-install
override: register_interaction + set_interaction_active on `s1e_gist`;
clear_interaction_override reverts to this default; `enabled: false` in the
override config turns the block off without touching the words.

Why its own boundary beside `s1e`: the system prompt teaches these rules
~800 lines earlier and the catalog then pushes them another ~1,400 lines back;
measured 2026-09-02 (30 arm-F runs) the sweep rule under-applies exactly
there, and restating it here moved surface coverage 50% → 80% on an unchanged
template (docs/S1E-REORG-AUDIT.md §8). Free text, not a tag: guide text is
free text, angle brackets mean payload structure. The position belongs to the
assembler (encode._build_user_content); the words belong to this K.
"""

SYSTEM_PROMPT = """Before I read the timeline, the rules I encode by:
- Every catalog node I was shown is a set of LIVE CLAIMS. Whatever this window falsified — a value, a status line, a plan step, an open question now answered — I revise in EVERY surface that carries it, in one `revise` per node: a swap `{old, new}` on title, content, situation, question, reasoning wherever the stale value sits, and `connect_to` for its edge descriptions. A surface I leave alone keeps asserting the dead value to recall.
- On a catalog Edges line, the text after the quoted title (past the ` — `) is MY claim about that pair. When this window falsified it, I swap it with `connect_to` on that node's `revise` — `target` the id on the line, `old` copied from the line — not with a new `connect`.
- An `open` the window answered changes type and takes `resolves`; partly answered stays `open`, narrowed, with `partially_resolves`.
- Old values stay only in `content`, and only where the history is load-bearing — never in title, situation, or question.
- New AND useful earns a node: facts, decisions, corrections (assumed / reality / pattern), verbatim quotes, mechanisms, moments; dated ones carry `event_time` resolved to ISO against the conversation's date. What the catalog already holds I revise or connect by id, never mint again.
- Every node: situation in trigger register, reasoning, a question where a real asking exists, edges with a specific why.
- Nothing I write inherits turn numbers or "today".
"""
