"""Shared metadata extractors used by multiple query modules.

Leading underscore by convention: these are package-internal helpers, not
public API. Nothing in `dashboard.server` should import from here directly —
go through the public `queries.X` modules that wrap them with their own
shaping.
"""

import json
from typing import Tuple


def extract_identity(metadata_raw) -> Tuple[str, str]:
    """Pull (human_identity, agent_identity) out of a trace's metadata JSON.

    Identity stamping (commits 75075eb / 65bf483 / 5cff407) records who was
    speaking when the trace was written. Both keys live inside the metadata
    JSON blob, NOT as separate columns — surfacing them as top-level fields
    is a UI concern.

    Returns ('', '') on any failure:
      - metadata is None / empty (trace pre-dates identity stamping)
      - metadata is not valid JSON
      - keys are absent (trace was written without identity configured)

    The dashboard treats absent identity as "unknown" rather than as an
    error — empty chips render as nothing.
    """
    return extract_meta_fields(metadata_raw, 'human_identity', 'agent_identity')


# The S0 session stamp — mirrors trace_contract.S0_SESSION_STAMP_FIELDS (the
# dashboard may not import servers.*; tests/test_s0_session_stamp.py holds the
# two literals in step). Which model produced the turn and which host runtime
# the stream rides on; per-turn on S0 rows, latest on the session row.
S0_SESSION_STAMP_FIELDS = ('model', 'host')


def extract_meta_fields(metadata_raw, *keys) -> tuple:
    """Pull `keys` out of a trace's metadata JSON as a tuple of strings, in
    order. '' for every key on any failure (no metadata, invalid JSON, key
    absent) — the dashboard treats absent as "unknown", never as an error."""
    empty = tuple('' for _ in keys)
    if not metadata_raw:
        return empty
    try:
        d = json.loads(metadata_raw)
    except (ValueError, TypeError):
        return empty
    if not isinstance(d, dict):
        return empty
    return tuple((d.get(k, '') or '') for k in keys)
