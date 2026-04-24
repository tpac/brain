"""Shared orientation for S1 scribe pipeline boundaries.

Both the S1 Scribe encoder (S1S) and the S1 Scouts receive similar structural
inputs: node catalog, surfaced-nodes-per-turn, conversation window, current
date. Before this module, each boundary had its own free-text explanation —
which drifted as the data format evolved.

This module is the SINGLE source of truth for what these fields MEAN, so
S1S and scouts read the same input the same way. When the render format
of any input changes (e.g. catalog rendering), the label here updates once
and both callers inherit.

## Layout

Per-section labels describe ONE field each (e.g. NODE_CATALOG_LABEL). Callers
compose their own preamble from the labels they actually receive — scouts
do NOT receive the encoding journal, for example.

Pre-composed blocks are provided for the two known callers:
    SCOUT_ORIENTATION_PREAMBLE  — scout framing + shared labels + scout closing
    S1S_WHAT_YOU_RECEIVE        — minimal header + shared labels incl. journal

Add new callers by composing the labels you need. Do NOT fork the labels —
diverging text in two places is the drift bug this module prevents.
"""
from __future__ import annotations


# ─── Per-section labels (single source of truth) ──────────────────────────

NODE_CATALOG_LABEL = """- **Node catalog** is what the brain ALREADY knows, pre-retrieved as relevant
  to this window. Each entry has id, title, content, situation, reasoning,
  metadata KV, and edges. Each node appears ONCE here, deduplicated across
  turns. Reference catalog nodes by id when a candidate or encoding relates
  to one."""

SURFACED_NODES_LABEL = """- **Surfaced nodes per turn** shows which catalog nodes the surfacer
  selected for that turn. Surfaced != used — a surfaced node was brought
  into the assistant's awareness but not necessarily referenced."""

CONVERSATION_WINDOW_LABEL = """- **Conversation window** is the last N turns of exchange. Each turn shows
  the operator message, the assistant response, and a list of surfaced node
  IDs for that turn. Surfaced IDs reference the catalog above — these are
  the nodes the surfacer selected to help the assistant remember relevant
  context when responding. Don't re-quote node content from the timeline."""

CURRENT_DATE_LABEL = """- **Current date** resolves relative time references."""

SESSION_CONTEXT_LABEL = """- **Session context** is the accumulated journey of this session
  (e.g. "dashboard fix | surfacer moved to daemon | encoder cleanup")."""

ENCODING_JOURNAL_LABEL = """- **Encoding journal** is what previous encoding runs captured, skipped,
  and flagged — your continuity within this session. Read before encoding
  so you don't re-evaluate topics the journal says you already handled."""


# ─── Composed blocks ──────────────────────────────────────────────────────

_SCOUT_SHARED_LABELS = "\n\n".join([
    SESSION_CONTEXT_LABEL,
    CURRENT_DATE_LABEL,
    NODE_CATALOG_LABEL,
    SURFACED_NODES_LABEL,
    CONVERSATION_WINDOW_LABEL,
])

_S1S_SHARED_LABELS = "\n\n".join([
    ENCODING_JOURNAL_LABEL,
    SESSION_CONTEXT_LABEL,
    NODE_CATALOG_LABEL,
    CONVERSATION_WINDOW_LABEL,
])


SCOUT_FRAMING = """## About this input

You are observing a conversation between the operator and the assistant.
The assistant runs with a persistent brain that encodes what matters every
few exchanges — you are part of that encoding process.

"""

SCOUT_CLOSING = """

Your job (below the cache break) is narrow. You surface candidates; the S1
Scribe composes the encoding. Scouts run in parallel and don't see each
other — some overlap between scout findings is expected, not a bug.
Do not write nodes."""


SCOUT_ORIENTATION_PREAMBLE = SCOUT_FRAMING + _SCOUT_SHARED_LABELS + SCOUT_CLOSING


S1S_WHAT_YOU_RECEIVE = """## What You Receive

""" + _S1S_SHARED_LABELS


__all__ = [
    # Per-section labels (single source of truth — compose freely)
    "NODE_CATALOG_LABEL",
    "SURFACED_NODES_LABEL",
    "CONVERSATION_WINDOW_LABEL",
    "CURRENT_DATE_LABEL",
    "SESSION_CONTEXT_LABEL",
    "ENCODING_JOURNAL_LABEL",
    # Scout-specific framing pieces (for callers that need the scout voice)
    "SCOUT_FRAMING",
    "SCOUT_CLOSING",
    # Pre-composed blocks for known callers
    "SCOUT_ORIENTATION_PREAMBLE",
    "S1S_WHAT_YOU_RECEIVE",
]
