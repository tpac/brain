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
    if not metadata_raw:
        return ('', '')
    try:
        d = json.loads(metadata_raw)
        return (d.get('human_identity', '') or '', d.get('agent_identity', '') or '')
    except (ValueError, TypeError):
        return ('', '')
